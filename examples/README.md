# Synthetic sales demo

`synthetic_sales_demo.xlsx` contains no original company or customer records. Upload it in the app's Data Penjualan tab. The first sheet, Sales, has exactly the import columns; About explains the design.

- 360 rows: six products × 60 months, January 2020–December 2024.
- Three recurring synthetic customers buy product pairs; suitable for exercising customer-month association analysis.
- Stable item codes and unique sales-ID/item-code pairs prevent accidental upsert collisions.
- Text dates match the importer's English month-name mapping.
- Quantity, price and discount contain fractional values so pandas reads float columns, as expected by the current importer. Quantity is in synthetic lots, including half-lots.
- Grand Total is an Excel formula: quantity × (unit price − per-lot discount), with cached values for import.
- All products appear in 2023 and 2024, with 60 monthly observations and 20 quarters.

Use a disposable local MongoDB instance. The importer persists data and does not automatically isolate this demo from existing records. Start with one forecast category and no external-variable workbook. This sample tests workflow compatibility, not forecasting accuracy or replication of thesis results.
