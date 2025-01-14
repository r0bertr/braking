import argparse
from dataclasses import MISSING, fields


def parse_args(args_cls: type):
    parser = argparse.ArgumentParser()
    for field in fields(args_cls):
        option = f"--{field.name.replace('_', '-')}"
        if field.default is MISSING and field.default_factory is MISSING:
            parser.add_argument(option, type=field.type, required=True)
        elif field.type is bool:
            if field.default is True:
                parser.add_argument(option, action="store_false")
            else:
                parser.add_argument(option, action="store_true")
        else:
            parser.add_argument(option, type=field.type,
                default=field.default if field.default is not MISSING
                    else field.default_factory())
    return args_cls(**parser.parse_args().__dict__)
