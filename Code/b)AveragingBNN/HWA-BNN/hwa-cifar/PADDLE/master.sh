rm $1.tar.gz
tar -czvf $1.tar.gz $1/

cd hadoop-v2/hadoop/bin/

sh removemount.sh /user/PUBLIC_KM_Data/v_karimibelhal/$1.tar.gz
sh mount.sh /home/belhal/belhal/$1.tar.gz /user/PUBLIC_KM_Data/v_karimibelhal

cd

cd belhal/