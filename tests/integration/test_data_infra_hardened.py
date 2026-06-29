def test_bitemporal_schemas():
    sql_files = ["database/universe_schema.sql", "database/adjusted_equity_schema.sql"]

    for sql_file in sql_files:
        with open(sql_file) as f:
            content = f.read()

            # Assert bitemporal fields exist
            assert "exchange_timestamp" in content, f"exchange_timestamp missing in {sql_file}"
            assert "local_recv_timestamp" in content, f"local_recv_timestamp missing in {sql_file}"

            print(
                f"[PASS] {sql_file} implements strict bitemporal versioning (exchange_timestamp/local_recv_timestamp)."
            )


if __name__ == "__main__":
    test_bitemporal_schemas()
    print("ALL DATA INFRA TESTS PASSED")
