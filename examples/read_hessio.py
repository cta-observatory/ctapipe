from ctapipe.io.hessio import hessio_event_source

import logging
logging.basicConfig(level=logging.DEBUG)

filename = "/Users/kosack/Data/CTA/Prod2/proton_20deg_180deg_run32364___cta-prod2_desert-1640m-Aar.simtel.gz"

for event in hessio_event_source(filename, max_events=10):
    print("=" * 70)
    print("EVENT_ID: ", event.event_id)
    print("TELS: ", event.tels_with_data)

    for tel in event.data:
        print("-" * 50)

        for chan in event.data[tel]:
            print("CT{:4d} ch{}".format(tel, chan))
            print(event.data[tel][chan])
