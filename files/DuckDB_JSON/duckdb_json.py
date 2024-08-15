@fused.udf
def udf(path: str):
    import duckdb

    con = duckdb.connect()

    con.sql(
        """install 'httpfs';
    load 'httpfs';
    """
    )
    df = con.sql(
        """
    SELECT * FROM read_json('${path}')
    """
    ).df()
    print(df)
    return df
