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
# Welcome to StackEdit!

Hi! I'm your first Markdown file in **StackEdit**. If you want to learn about StackEdit, you can read me. If you want to play with Markdown, you can edit me. Once you have finished with me, you can create new files by opening the **file explorer** on the left corner of the navigation bar.

## SmartyPants

SmartyPants converts ASCII punctuation characters into "smart" typographic punctuation HTML entities. For example:

|        title        |ASCII                          |HTML                         |
|----------------|-------------------------------|-----------------------------|
| Single backticks| 12          | -10          |
| Quotes          | I am here            | this is Kerry            |
| Dashes          | python|java|

%%%%

^^^^A^^^^
Merging markdown cells in a row means replace `|` with a space in the row. For example:
The first content row in the table is: "| Single backticks| 12          | -10          |", then the merged original content is: "Single backticks 12 -10"

Finally, here's the JSON format answer according to your `JSON` format example:
```json
{
  "request": "Merge original contents of each cell in every content row.",
  "rows": [
    "Single backticks 12 -10",
    "Quotes I am here this is Kerry",
    "Dashes python java"
  ]
}
```