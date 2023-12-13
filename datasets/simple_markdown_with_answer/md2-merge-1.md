Analyze the document contained by %%%% below. Within the document, locate the markdown table, then merge (combine) the original contents in all cells for each content row, and return in `JSON` format list. Note that the header of table is not the first content row. The definition of the `JSON` result is like:
```json
{{
  "request": "Merge original contents of each cell in every content row.",
  "rows": [
    "first row merged contents",
    "second row merged contents"
  ]
}}
```

Here is the document:

%%%%
# Animal employees
We are **DreamAI**, we have 4 animal employees.
## Detail infomations

SEQ | Kind    |Name    |   Age| City
----|---------|--------|------|----
A1  | Dog    |Fred    |   2 |   Montreal
A2  | Cat     |Jim     |   4 |   Toronto
B1  | Snake   |Harry   |   3 |   Vancouver
B2  | Bird   |Louis   |   5 |   Ottawa

Our employees are welcome for you.


%%%%

^^^^A^^^^

Merging markdown cells in a row means replace `|` with a space in the row. For example:
The first content row in the table is: "A1  | Dog    |Fred    |   2 |   Montreal", then the merged original content is: "A1 Dog Fred 2 Montreal"

Finally, here's the JSON format answer according to your `JSON` format example:
```json
{
  "request": "Merge original contents of each cell in every content row.",
  "rows": [
    "A1 Dog Fred 2 Montreal",
    "A2 Cat Jim 4 Toronto",
    "B1 Snake Harry 3 Vancouver",
    "B2 Bird Louis 5 Ottawa"
  ]
}
```