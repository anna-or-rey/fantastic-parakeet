# app/tools/fx.py
from semantic_kernel.functions import kernel_function
import requests
import json
from app.utils.logger import setup_logger

logger = setup_logger("fx_tool")

class FxTools:
    @kernel_function(name="convert_fx", description="Convert currency via Frankfurter API")
    def convert_fx(self, amount: float, base: str, target: str) -> str:
        """
        Convert currency using the Frankfurter API.

        Args:
            amount: Amount to convert
            base: Source currency code (e.g., USD)
            target: Target currency code (e.g., EUR)

        Returns:
            JSON string with conversion result including rate and converted amount
        """
        logger.info(f"FX tool called: {amount} {base} -> {target}")

        try:
            # Normalize currency codes to uppercase
            base = base.upper().strip()
            target = target.upper().strip()

            # Make API request to Frankfurter
            url = "https://api.frankfurter.app/latest"
            params = {
                "amount": amount,
                "from": base,
                "to": target
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract the converted amount
            rates = data.get("rates", {})
            converted_amount = rates.get(target)

            if converted_amount is None:
                logger.warning(f"Currency {target} not found in response")
                return json.dumps({
                    "error": f"Currency '{target}' not supported",
                    "base": base,
                    "target": target,
                    "amount": amount,
                    "converted_amount": None,
                    "rate": None
                })

            # Calculate the exchange rate
            rate = converted_amount / amount if amount > 0 else 0

            result = {
                "base": base,
                "target": target,
                "amount": amount,
                "converted_amount": round(converted_amount, 2),
                "rate": round(rate, 6),
                "date": data.get("date", "unknown")
            }

            logger.info(f"FX result: {amount} {base} = {converted_amount:.2f} {target} (rate: {rate:.4f})")
            return json.dumps(result)

        except requests.RequestException as e:
            logger.error(f"FX API request failed: {e}")
            return json.dumps({
                "error": f"Currency conversion failed: {str(e)}",
                "base": base,
                "target": target,
                "amount": amount,
                "converted_amount": None,
                "rate": None
            })
        except Exception as e:
            logger.error(f"FX tool error: {e}")
            return json.dumps({
                "error": f"FX tool error: {str(e)}",
                "base": base,
                "target": target,
                "amount": amount,
                "converted_amount": None,
                "rate": None
            })
