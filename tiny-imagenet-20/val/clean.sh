find . -type d -depth 2 | while read -r dir; do
  find "$dir" -type f -maxdepth 1 | head -n 50 > "$dir/.keep"
  find "$dir" -type f -maxdepth 1 | grep -v -F -f "$dir/.keep" | while read -r file; do
    rm "$file"
  done
  rm "$dir/.keep"
done

