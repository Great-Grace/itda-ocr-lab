import paddle
for key in ['FLAGS_use_mkldnn','FLAGS_use_onednn','FLAGS_enable_pir_api']:
    try: print(key, paddle.get_flags([key]))
    except Exception as e: print(key, repr(e))
