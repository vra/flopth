""" Settings for flopth. """


class Settings:
    image_height = 224
    image_width = 224

    # parameter dict of info
    param_dict = {
        'flops': {
            'text': 'FLOPs',
            'size': 1,
            'type': 'long'
        },
    }


settings = Settings()
