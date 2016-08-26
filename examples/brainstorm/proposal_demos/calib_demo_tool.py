from calib_demo_flow import *
import logging
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pedestal Calibration Demo")

    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                          argparse.RawDescriptionHelpFormatter):
        pass

    parser.formatter_class = CustomFormatter
    parser.add_argument('-l', '--loglevel',
                        choices=['info', 'debug', 'warning'],
                        default="info", help="verbosity to write to log")
    parser.add_argument('--chunksize', metavar='N',
                        type=int, default=300,
                        help="number of events per chunk")
    parser.add_argument('--chunklimit', metavar='N',
                        type=int, default=None,
                        help="number of chunks to generate")
    parser.add_argument('-d', '--display', action='store_true',
                        help="enable display")

    opts = parser.parse_args()

    # set log level
    logging.basicConfig(level=getattr(logging, opts.loglevel.upper()))
    logging.info("OPTIONS:")
    logging.info(opts)

    # define the chain
    chain = (
        gen_fake_raw_images(chunksize=opts.chunksize,
                            chunklimit=opts.chunklimit)
        | calc_and_apply_pedestals
        | check_results
        | show_throughput(every=10, events_per_cycle=opts.chunksize,
                          identifier="PED")
        | tee_every(every=100)
    )

    if opts.display:
        chain |= display_pedvars(every=100)

    try:
        for data in chain:
            pass
    except Exception as err:
        logging.warn(err)
    finally:
        logging.info("finishing")


