for file in input_files/ANLB_p11/*.inp; do
    uv run python main.py "$file"
done