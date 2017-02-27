WID=`xdotool search "Mozilla Firefox" `;
xdotool windowactivate --sync $WID;
sleep 1;
filename=$1
#echo $filename
count=1
passstr=""
while read p; do
    if [ $count -lt 40 ]
    then
    	passstr=$passstr" "$p
	#echo $passstr
 	count=$(( $count + 1 ))
    else
	#echo $passstr
	xdotool type $passstr
	sleep 1
	count=1
	passstr=" "
    fi
done < $filename
xdotool type $passstr
#xdotool type "hello world ";
sleep 2;
xdotool key Tab; 
xdotool key Return
