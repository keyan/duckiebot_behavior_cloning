read -r -p "Download original datasets? [y/N] " response
case "$response" in
  [yY])
    # Fetches the current best dataset, a mix of simulated and real robot images, about 5000 frames.
    wget https://www.dropbox.com/s/ztari0lwvz51cbd/simulated_plus_maserati.log?dl=1 -O simulated_plus_maserati.log

    # This is a very small dataset which is good for testing new models locally for syntax errors/etc.
    wget https://www.dropbox.com/s/03zlt4m4ug5nx4r/sim_small_plus_tori.log?dl=1 -O sim_small_plus_tori.log

    # huge dataset, mostly simulated data
    wget https://www.dropbox.com/s/ms75dkdin6xr89x/maserati_bill_simulated_amadobot_base.log?dl=1 -O maserati_bill_simulated_amadobot_base.log
    ;;
  *)
    ;;
esac

read -r -p "Download custom robot datasets? [y/N] " response
case "$response" in
  [yY])
    # Custom dataset recorded using human driver on Keyan's robot. Includes data from traveling
    # in counter-clockwise direction on the track only.
    wget https://www.dropbox.com/s/6j99x9wqe6qtjb4/alfredo_ccw.log?dl=1 -O alfredo_ccw.log

    # Custom dataset recorded using human driver on Keyan's robot. Same as above dataset but also
    # includes data from driving clockwise direction on the track.
    wget https://www.dropbox.com/s/444pr9dusd74n54/alfredo_ccw_plus_cw.log?dl=1 -O alfredo_ccw_plus_cw.log
    ;;
  *)
    ;;
esac
