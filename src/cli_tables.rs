use cli_table::{format::Justify, print_stdout, Cell, CellStruct, Style, Table, TableStruct};
use colored::Colorize;

/// Prints a table to the console.
fn print_table(table: TableStruct) {
    print_stdout(table).unwrap();
}

/// Prints a title to the console.
pub fn print_title(title: &str) {
    let table = vec![vec![title
        .green()
        .cell()
        .bold(true)
        .justify(Justify::Center)]]
    .table();
    print_table(table);
}

/// Prints the table with the runtime arguments to the console.
pub fn print_args_table(elements: Vec<Vec<CellStruct>>) {
    let table = elements.table().title(vec![
        "Argument".cell().bold(true),
        "Value".cell().bold(true),
    ]);
    print_table(table);
}
