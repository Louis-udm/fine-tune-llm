Analyze the document contained by %%%% below. Within the document, locate the markdown table, then merge (combine) the original contents row by row, and return in `JSON` format list. Note do not include the header of the table. 

Here is the document:

%%%%
# Welcome to StackEdit!

Hi! The **file explorer** is on the left corner of the navigation bar.

## SmartyPants

SmartyPants converts ASCII punctuation characters into "smart" typographic punctuation HTML entities. For example:

|        title        |ASCII                          |HTML                         |
|----------------|-------------------------------|-----------------------------|
| Single backticks| 12          | -10          |
| Quotes          | I am here            | this is Kerry            |
| Dashes          | python|java|

%%%%

^^^^A^^^^
Here you are the merged rows in `JSON` format:

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