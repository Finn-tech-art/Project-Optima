from __future__ import annotations

import base64
import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import httpx

from backend.core.exceptions import (
    KrakenAPIResponseError,
    KrakenAPIUnavailableError,
)


KRAKEN_PUBLIC_PREFIX = "/0/public"
KRAKEN_PRIVATE_PREFIX = "/0/private"


# This block defines the runtime configuration for Kraken REST access.
# It takes: the API credentials, base URL, and request/validation settings.
# It gives: one normalized config object for authenticated Kraken access.
@dataclass(slots=True, frozen=True)
class KrakenRESTConfig:
    api_key: str
    api_secret: str
    base_url: str = "https://api.kraken.com"
    timeout_seconds: float = 5.0
    validate_only: bool = True


# This block stores normalized ticker values used by market and execution logic.
# It takes: the raw Kraken ticker payload for one pair.
# It gives: a small typed snapshot of bid/ask/last-trade values.
@dataclass(slots=True, frozen=True)
class KrakenTickerSnapshot:
    pair: str
    altname: str
    bid: float
    ask: float
    last: float | None
    raw: dict[str, Any]


# This block is the main Kraken REST adapter.
# It takes: a validated REST config and handles public/private Kraken API calls.
# It gives: market, balance, and order helpers for live runtime providers.
@dataclass(slots=True)
class KrakenRESTClient:
    config: KrakenRESTConfig
    _client: httpx.Client = field(init=False, repr=False)
    _pairs_cache: dict[str, dict[str, Any]] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.config.base_url.rstrip("/"),
            timeout=self.config.timeout_seconds,
        )

    # This block checks whether Kraken's public API is currently reachable.
    # It takes: no runtime input beyond the configured HTTP client.
    # It gives: a simple boolean readiness signal for market/runtime bootstrapping.
    def ping(self) -> bool:
        try:
            self.get_asset_pairs(force_refresh=True)
            return True
        except Exception:  # noqa: BLE001
            return False

    # This block fetches Kraken tradable pair metadata and caches it.
    # It takes: an optional refresh flag.
    # It gives: the full AssetPairs result keyed by Kraken pair id.
    def get_asset_pairs(self, *, force_refresh: bool = False) -> dict[str, dict[str, Any]]:
        if self._pairs_cache is not None and not force_refresh:
            return self._pairs_cache

        payload = self._request_public("/AssetPairs")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise KrakenAPIResponseError(
                "Kraken AssetPairs response did not include a result mapping.",
                context={"payload": payload},
            )

        self._pairs_cache = result
        return result

    # This block resolves an internal symbol like BTCUSD into Kraken pair metadata.
    # It takes: a repo-native symbol string.
    # It gives: the matching Kraken pair id and metadata needed for later requests.
    def resolve_pair(self, symbol: str) -> tuple[str, dict[str, Any]]:
        normalized_symbol = self._normalize_symbol(symbol)
        pairs = self.get_asset_pairs()

        for pair_id, metadata in pairs.items():
            if not isinstance(metadata, dict):
                continue

            candidates = {
                self._normalize_symbol(pair_id),
                self._normalize_symbol(str(metadata.get("altname", ""))),
                self._normalize_symbol(str(metadata.get("wsname", ""))),
            }

            if normalized_symbol in candidates:
                return pair_id, metadata

        raise KrakenAPIResponseError(
            "Unable to resolve the requested symbol to a Kraken trading pair.",
            context={"symbol": symbol},
        )

    # This block fetches the latest public ticker for a symbol.
    # It takes: a repo-native symbol string like BTCUSD.
    # It gives: a typed bid/ask snapshot plus the raw pair payload.
    def get_ticker(self, symbol: str) -> KrakenTickerSnapshot:
        pair_id, metadata = self.resolve_pair(symbol)
        altname = str(metadata.get("altname", pair_id))
        payload = self._request_public("/Ticker", params={"pair": altname})

        result = payload.get("result")
        if not isinstance(result, dict) or not result:
            raise KrakenAPIResponseError(
                "Kraken ticker response did not include result data.",
                context={"symbol": symbol, "payload": payload},
            )

        raw_pair_payload = next(iter(result.values()))
        if not isinstance(raw_pair_payload, dict):
            raise KrakenAPIResponseError(
                "Kraken ticker result for the resolved pair was malformed.",
                context={"symbol": symbol, "payload": payload},
            )

        bid = self._extract_price_field(raw_pair_payload, "b")
        ask = self._extract_price_field(raw_pair_payload, "a")
        last = self._extract_optional_price_field(raw_pair_payload, "c")

        return KrakenTickerSnapshot(
            pair=pair_id,
            altname=altname,
            bid=bid,
            ask=ask,
            last=last,
            raw=raw_pair_payload,
        )

    # This block fetches authenticated balance data from Kraken.
    # It takes: no runtime input beyond configured credentials.
    # It gives: the raw balance mapping returned by Kraken.
    def get_balances(self) -> dict[str, str]:
        payload = self._request_private("/Balance")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise KrakenAPIResponseError(
                "Kraken balance response did not include a result mapping.",
                context={"payload": payload},
            )
        return {
            str(asset): str(amount)
            for asset, amount in result.items()
        }

    # This block fetches authenticated open-orders data from Kraken.
    # It takes: no runtime input beyond configured credentials.
    # It gives: the raw open-order mapping used for portfolio pressure checks.
    def get_open_orders(self) -> dict[str, Any]:
        payload = self._request_private("/OpenOrders")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise KrakenAPIResponseError(
                "Kraken open-orders response did not include a result mapping.",
                context={"payload": payload},
            )

        open_orders = result.get("open", {})
        if not isinstance(open_orders, dict):
            raise KrakenAPIResponseError(
                "Kraken open-orders result did not include a valid 'open' mapping.",
                context={"payload": payload},
            )
        return open_orders

    # This block places an authenticated order on Kraken.
    # It takes: a symbol, side, order type, volume, and optional limit price.
    # It gives: the raw Kraken AddOrder result for execution reconciliation.
    def add_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        volume: float,
        validate_only: bool | None = None,
        price: float | None = None,
        time_in_force: str | None = None,
    ) -> dict[str, Any]:
        pair_id, metadata = self.resolve_pair(symbol)
        altname = str(metadata.get("altname", pair_id))
        payload: dict[str, Any] = {
            "pair": altname,
            "type": side.lower(),
            "ordertype": order_type.lower(),
            "volume": self._format_decimal(volume),
            "validate": validate_only if validate_only is not None else self.config.validate_only,
        }

        if price is not None and order_type.lower() == "limit":
            payload["price"] = self._format_decimal(price)

        if time_in_force:
            payload["timeinforce"] = time_in_force.lower()

        response = self._request_private("/AddOrder", data=payload)
        result = response.get("result")
        if not isinstance(result, dict):
            raise KrakenAPIResponseError(
                "Kraken AddOrder response did not include a result mapping.",
                context={"symbol": symbol, "payload": response},
            )
        return result

    # This block issues a public Kraken REST request.
    # It takes: the endpoint path and optional query parameters.
    # It gives: the decoded Kraken response payload after error handling.
    def _request_public(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request("GET", f"{KRAKEN_PUBLIC_PREFIX}{path}", params=params)

    # This block issues a signed private Kraken REST request.
    # It takes: the endpoint path and form payload.
    # It gives: the decoded Kraken response payload after auth and error handling.
    def _request_private(
        self,
        path: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = dict(data or {})
        payload["nonce"] = self._build_nonce()
        body = urlencode(payload)
        uri_path = f"{KRAKEN_PRIVATE_PREFIX}{path}"
        signature = self._sign(uri_path=uri_path, body=body, nonce=str(payload["nonce"]))
        headers = {
            "API-Key": self.config.api_key,
            "API-Sign": signature,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }
        return self._request("POST", uri_path, headers=headers, content=body)

    # This block performs the actual HTTP request and enforces Kraken error semantics.
    # It takes: the method, URI path, and optional request parameters.
    # It gives: a decoded JSON payload or a structured Kraken API error.
    def _request(
        self,
        method: str,
        uri_path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        content: str | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(
                method=method,
                url=uri_path,
                params=params,
                headers=headers,
                content=content,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise KrakenAPIResponseError(
                "Kraken API returned an error status.",
                context={
                    "status_code": error.response.status_code,
                    "uri_path": uri_path,
                    "response_text": error.response.text,
                },
            ) from error
        except httpx.HTTPError as error:
            raise KrakenAPIUnavailableError(
                "Kraken API request failed due to a transport error.",
                context={"uri_path": uri_path, "error": str(error)},
            ) from error

        try:
            payload = response.json()
        except ValueError as error:
            raise KrakenAPIResponseError(
                "Kraken API did not return valid JSON.",
                context={"uri_path": uri_path, "response_text": response.text},
            ) from error

        if not isinstance(payload, dict):
            raise KrakenAPIResponseError(
                "Kraken API returned an unexpected payload type.",
                context={"uri_path": uri_path, "payload_type": type(payload).__name__},
            )

        error_list = payload.get("error", [])
        if isinstance(error_list, list) and error_list:
            raise KrakenAPIResponseError(
                "Kraken API reported one or more request errors.",
                context={"uri_path": uri_path, "errors": error_list, "payload": payload},
            )

        return payload

    # This block generates the Kraken signature for a private request.
    # It takes: the private URI path, form-encoded body, and nonce string.
    # It gives: the base64-encoded API-Sign header value.
    def _sign(self, *, uri_path: str, body: str, nonce: str) -> str:
        nonce_plus_postdata = f"{nonce}{body}".encode("utf-8")
        sha256_hash = hashlib.sha256(nonce_plus_postdata).digest()
        message = uri_path.encode("utf-8") + sha256_hash
        secret = base64.b64decode(self.config.api_secret)
        signature = hmac.new(secret, message, hashlib.sha512).digest()
        return base64.b64encode(signature).decode("utf-8")

    # This block builds the monotonically increasing nonce used by private calls.
    # It takes: no runtime input.
    # It gives: a millisecond precision nonce string acceptable to Kraken.
    def _build_nonce(self) -> str:
        return str(int(time.time() * 1000))

    # This block normalizes a symbol into an exchange-agnostic comparison string.
    # It takes: a symbol or pair identifier from repo or Kraken metadata.
    # It gives: a stripped uppercase symbol suitable for loose pair matching.
    def _normalize_symbol(self, value: str) -> str:
        normalized = value.upper().replace("/", "").replace("-", "").replace("_", "")
        return normalized.replace("XBT", "BTC")

    # This block extracts the first price value from a Kraken list-style price field.
    # It takes: the raw pair payload and target field name like 'a' or 'b'.
    # It gives: a float price value for downstream market/execution logic.
    def _extract_price_field(self, payload: dict[str, Any], field_name: str) -> float:
        value = payload.get(field_name)
        if isinstance(value, list) and value:
            return float(value[0])
        raise KrakenAPIResponseError(
            "Kraken ticker response is missing a required price field.",
            context={"field_name": field_name, "payload": payload},
        )

    # This block extracts an optional last-trade field from a Kraken ticker payload.
    # It takes: the raw pair payload and target field name.
    # It gives: a float price when present, otherwise None.
    def _extract_optional_price_field(
        self,
        payload: dict[str, Any],
        field_name: str,
    ) -> float | None:
        value = payload.get(field_name)
        if isinstance(value, list) and value:
            return float(value[0])
        return None

    # This block formats decimals for Kraken form submissions.
    # It takes: a float volume or price.
    # It gives: a string without unnecessary scientific notation.
    def _format_decimal(self, value: float) -> str:
        return format(value, ".10f").rstrip("0").rstrip(".")
