"""
This module provides functionality for creating and managing tables.
"""

import pandas as pd
from typing import List, Optional

class Table:
    def __init__(self, title: str = "", schema: list = [], description: str = ""):
        """
        Initialize the Table object.
        
        Args:
        title (str): The title of the table.
        schema (list): List of column names.
        description (str): Description of the table.
        """
        self.title = title
        self.column_names = schema  # Store column headers
        self.table = []  # Store rows of the table
        self.primary_key = None
        self.description = description

    def create_table(self, title: str = "", schema: List[str] = [], rows: List[List[str]] = [], description: str = "") -> None:
        """
        Create a new table with given title, schema, and rows.
        
        Args:
        title (str): The title of the table.
        schema (list): List of column names.
        rows (list): List of rows to populate the table.
        description (str): Description of the table.
        """
        self.title = title
        self.column_names = schema
        self.description = description
        for row in rows:
            self.add_row(row)

    def set_columns(self, columns: List[str]) -> None:
        """
        Set the column names of the table.
        
        Args:
        columns (list): List of new column names.
        """
        self.column_names = columns

    def add_row(self, row: List[str]) -> None:
        """
        Add a new row to the table.
        
        Args:
        row (list): List of values for the row. The length of the row must match the number of columns.
        
        Raises:
        ValueError: If the row length does not match the number of columns.
        """
        if len(row) != len(self.column_names):
            raise ValueError("Row length must match the number of columns")
        self.table.append(row)

    def get_rows(self) -> List[List[str]]:
        """
        Get all the rows of the table.
        
        Returns:
        list: List of rows in the table.
        """
        return self.table

    def get_columns(self) -> List[str]:
        """
        Get all the column names of the table.
        
        Returns:
        list: List of column names.
        """
        return self.column_names

    def get_column_data(self, column_name: str) -> List[str]:
        """
        Get the data of a specific column by its name.
        
        Args:
        column_name (str): The name of the column whose data is to be retrieved.
        
        Returns:
        list: List of values in the specified column.
        
        Raises:
        ValueError: If the column name is not found.
        """
        if column_name not in self.column_names:
            raise ValueError("Column not found")
        index = self.column_names.index(column_name)
        return [row[index] for row in self.table]

    def get_column_data_by_index(self, index: int) -> List[str]:
        """
        Get the data of a specific column by its index.
        
        Args:
        index (int): The index of the column whose data is to be retrieved.
        
        Returns:
        list: List of values in the specified column.
        
        Raises:
        IndexError: If the index is out of range.
        """
        if index < 0 or index >= len(self.column_names):
            raise IndexError("Column index out of range")
        return [row[index] for row in self.table]

    def delete_table(self) -> None:
        """
        Delete all the rows and columns of the table.
        """
        self.table.clear()
        self.column_names.clear()

    def delete_row(self, index: int) -> None:
        """
        Delete a specific row by its index.
        
        Args:
        index (int): The index of the row to be deleted.
        
        Raises:
        IndexError: If the index is out of range.
        """
        if index < 0 or index >= len(self.table):
            raise IndexError("Row index out of range")
        self.table.pop(index)

    def delete_column(self, column_name: str) -> None:
        """
        Delete a specific column by its name.
        
        Args:
        column_name (str): The name of the column to be deleted.
        
        Raises:
        ValueError: If the column name is not found.
        """
        if column_name not in self.column_names:
            raise ValueError("Column not found")
        index = self.column_names.index(column_name)
        self.delete_column_by_index(index)

    def delete_column_by_index(self, index: int) -> None:
        """
        Delete a specific column by its index.
        
        Args:
        index (int): The index of the column to be deleted.
        
        Raises:
        IndexError: If the index is out of range.
        """
        if index < 0 or index >= len(self.column_names):
            raise IndexError("Column index out of range")
        self.column_names.pop(index)
        for row in self.table:
            row.pop(index)

    def add_subtable(self, subtable: 'Table') -> None:
        """
        Add a subtable to the current table.
        
        Args:
        subtable (Table): The subtable to be added.
        
        Raises:
        ValueError: If the column names do not match between the tables.
        """
        if self.column_names != subtable.get_columns():
            raise ValueError("Column names must match to merge tables")
        self.table.extend(subtable.get_rows())
        self.description += f"\nMerged with subtable: {subtable.description}"

    def delete_subtable(self, subtable: 'Table') -> None:
        """
        Delete a subtable from the current table.
        
        Args:
        subtable (Table): The subtable to be deleted.
        """
        subtable_rows = subtable.get_rows()
        self.table = [row for row in self.table if row not in subtable_rows]
        self.description = self.description.replace(f"\nMerged with subtable: {subtable.description}", "")

    def visualize(self, filename: str) -> str:
        """
        Visualize the table as a pandas DataFrame.
        
        Returns:
        str: A Markdown format of the table as a string.
        """
        df = pd.DataFrame(self.table, columns=self.column_names)
        markdown_text = df.to_markdown(index=False)  # no index columna
        with open(filename, "w") as file:
            file.write(markdown_text)
        return markdown_text
    

    def natural_merge(self, other: 'Table', join_type: str = "outer") -> 'Table':
        """
        Perform a natural join between the current table and another table, returning a new Table instance.

        Args:
            other (Table): Another Table object.
            join_type (str): "outer" or "inner", default is outer join.

        Returns:
            Table: The merged new Table instance.

        Raises:
            ValueError: If there are no common columns or the join_type is invalid.
        """
        assert join_type in ["outer", "inner"], "join_type must be 'outer' or 'inner'"

        # 转换为 pandas.DataFrame
        df1 = pd.DataFrame(self.table, columns=self.column_names)
        df2 = pd.DataFrame(other.table, columns=other.column_names)

        # 清理列名中的 < >
        df1.columns = [c.strip().strip("<>") for c in df1.columns]
        df2.columns = [c.strip().strip("<>") for c in df2.columns]

        # 找出公共列
        common_cols = list(set(df1.columns).intersection(set(df2.columns)))
        if not common_cols:
            raise ValueError(f"❌ Cannot perform natural join, no common columns between the two tables")

        # 执行 merge
        merged = pd.merge(df1, df2, on=common_cols, how=join_type, suffixes=('', '_dup'))

        # 合并 _dup 列
        for col in merged.columns:
            if col.endswith('_dup'):
                base = col[:-4]
                merged[base] = merged[base].combine_first(merged[col])
                merged.drop(columns=[col], inplace=True)

        # 创建新的 Table 实例
        merged_table = Table(
            title=f"{self.title} + {other.title}",
            schema=list(merged.columns),
            description=f"Merged from:\n- {self.description}\n- {other.description}"
        )
        for _, row in merged.iterrows():
            merged_table.add_row(list(row))

        return merged_table
    
    @staticmethod
    def natural_merge_many(tables: List['Table'], join_type: str = "outer") -> 'Table':
        """
        Batch natural join multiple Table objects, returning a merged new Table.

        Args:
            tables (List[Table]): The list of Table objects to merge.
            join_type (str): "outer" or "inner", default is outer.

        Returns:
            Table: The merged new Table object.

        Raises:
            ValueError: If the table list is empty or there is no common column.
        """
        assert join_type in ["outer", "inner"], "join_type must be 'outer' or 'inner'"
        if not tables:
            raise ValueError("The table list cannot be empty")

        # 初始化第一个表为起点
        base_table = tables[0]

        for next_table in tables[1:]:
            base_table = base_table.natural_merge(next_table, join_type=join_type)

        return base_table




if __name__ == "__main__":
    # Example usage
    data = [
        ["Alice", 30, "HR"],
        ["Bob", 25, "Engineering"],
        ["Charlie", 35, "Finance"]
    ]

    # Instantiate a table object
    main_table = Table()
    main_table.create_table(title="Employee data", schema=["Name", "Age", "Department"], rows=data, description="Main table containing employee data")

    # Add a new data row
    main_table.add_row(["Simon", 20, "Tech"])

    # Retrieve all rows of data
    print("All rows:", main_table.get_rows())

    # Retrieve column names
    print("Column names:", main_table.get_columns())

    # Retrieve data from a specific column by column name
    print("Ages:", main_table.get_column_data("Age"))

    # # Delete a row by index
    # main_table.delete_row(1)  # Delete the row with index 1 (Bob)

    # # Delete a column by column name
    # main_table.delete_column("Department")

    # # Display the table data after deletions
    # print("After deletions:", main_table.get_rows())

    # Create a subtable
    sub_table = Table(description="Subtable containing new employee data")
    sub_table.set_columns(["Name", "Age", "Department"])
    sub_table.add_row(["Diana", 28, "Marketing"])
    sub_table.add_row(["Eve", 22, "Sales"])

    # Merge the subtable into the main table
    main_table.add_subtable(sub_table)

    # Display the table data after merging the subtable
    print("After merging subtable:", main_table.get_rows())

    # Delete the subtable data
    main_table.delete_subtable(sub_table)

    # Display the table data after deleting the subtable
    print("After deleting subtable:", main_table.get_rows())
