for file in ANLB_p11/*.inp; do
    python3 main.py "$file"
done
mv output ANLB_p11/Output