FILE_ID="1sw9gdk_igiyyt7MqALyxZhRrtPvAn0sX"
CONFIRM=$(curl -c /tmp/cookies.txt "https://drive.google.com/uc?export=download&id=${FILE_ID}" | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
curl -Lb /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" -o ~/scratch/cache/shapenet.zip
rm -f /tmp/cookies.txt

