**The big idea:** When a planet passes in front of its star, ==🔴the star gets slightly dimmer for a short time==. Telescopes like Kepler measured the brightness of about 150,000 stars for years. ==🔴If the dip repeats on a regular schedule, it could be a planet.==

**The problem:** ==🔴Not every dip is a planet.== Some come from *two stars orbiting each other, from instrument glitches, or from plain noise.* Astronomers have thousands of these signals to check, and they can’t inspect each one by hand.

**Objective** - Monitor the star's brightness over a time period to decide a planet candidate or false alarm.


TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP"

columns fetched from DR24 TCE Table 
COLS = [
    # identity and split key
    "kepid", "tce_plnt_num",
    # ephemeris for detrending and folding
    "tce_period",     # days
    "tce_time0bk",    # days, BKJD
    "tce_duration",   # HOURS
    # label source
    "av_training_set",
    # reference values
    "tce_depth",      # ppm
    "tce_model_snr",
    # useful for the download step and filtering
    "tce_quarters",
    "tce_num_transits",
]

parquet - binary format file for efficient acess