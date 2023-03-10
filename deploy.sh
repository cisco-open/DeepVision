while getopts b: flag
do
    case "${flag}" in
        b) branch=${OPTARG};;
    esac
done
echo "Deploying from Branch: $branch .."

sudo cp -R CiscoDeepVision/ CiscoDeepVision_bkp
sudo rm -rf CiscoDeepVision
git clone -b $branch git@github.com:CiscoDeepVision/CiscoDeepVision.git CiscoDeepVision

echo "Successfully deployed from branch $branch to server"


