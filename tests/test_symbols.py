from quant_lab.symbols import normalize_symbol


def test_normalize_symbol_variants_to_okx_style():
    assert normalize_symbol("BNB/USDT") == "BNB-USDT"
    assert normalize_symbol("BNB-USDT") == "BNB-USDT"
    assert normalize_symbol("BNBUSDT") == "BNB-USDT"
    assert normalize_symbol("OKX:BNB-USDT") == "BNB-USDT"
