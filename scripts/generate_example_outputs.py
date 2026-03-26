from tsyparty.cli import cmd_example, build_parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args(["example"])
    args.func(args)
