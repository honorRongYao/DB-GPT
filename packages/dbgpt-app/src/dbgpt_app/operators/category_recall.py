"""使用 Doris 的 bge_embed 和 L2_DISTANCE 召回关键词并生成来源表 SQL。"""

import os
from typing import Dict, List, Set, Tuple

import pymysql


DORIS_HOST = os.getenv("DORIS_HOST", "10.233.3.31")
DORIS_PORT = int(os.getenv("DORIS_PORT", "9030"))
DORIS_USER = os.getenv("DORIS_USER", "root")
DORIS_PASSWORD = os.getenv("DORIS_PASSWORD", "")
DATABASE = "voc_ai_test"
TARGET_TABLE = "category_embedding"
EMBEDDING_FUNCTION = "voc.bge_embed"
DEFAULT_MAX_DISTANCE = 0.8


def create_connection():
    return pymysql.connect(
        host=DORIS_HOST,
        port=DORIS_PORT,
        user=DORIS_USER,
        password=DORIS_PASSWORD,
        database=DATABASE,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_identifier(value: str) -> str:
    return "`" + value.replace("`", "``") + "`"


def build_distance_condition(query: str, max_distance: float) -> str:
    return (
        f"L2_DISTANCE(`vector`, {EMBEDDING_FUNCTION}({sql_quote(query)})) "
        f"< {max_distance}"
    )


def build_source_sql(
    query: str,
    max_distance: float,
    table_name: str,
    key_word: str,
    column_names: List[str],
) -> str:
    condition = build_distance_condition(query, max_distance)
    column_literals = ", ".join(sql_quote(name) for name in column_names)
    join_conditions = " OR\n    ".join(
        f"t1.{quote_identifier(name)} = t2.`raw_data`" for name in column_names
    )
    return "\n".join(
        [
            f"# 关键词:{query} 来自表 {table_name}，主键 {key_word}，命中列 {', '.join(column_names)}",
            "SELECT DISTINCT",
            f"    t1.{quote_identifier(key_word)}",
            f"FROM {quote_identifier(DATABASE)}.{quote_identifier(table_name)} AS t1",
            "LEFT JOIN (",
            "    SELECT DISTINCT `raw_data`",
            f"    FROM `{DATABASE}`.`{TARGET_TABLE}`",
            f"    WHERE {condition}",
            f"      AND `table_name` = {sql_quote(table_name)}",
            f"      AND `column_name` IN ({column_literals})",
            ") AS t2",
            "ON " + join_conditions,
            "WHERE t2.`raw_data` IS NOT NULL;",
        ]
    )


def recall(query: str, max_distance: float = DEFAULT_MAX_DISTANCE) -> Dict:
    query = query.strip()
    if not query:
        raise ValueError("query 不能为空")
    if max_distance < 0:
        raise ValueError("max_distance 不能小于 0")

    condition = build_distance_condition(query, max_distance)
    discovery_sql = (
        "SELECT DISTINCT `table_name`, `key_word`, `column_name` "
        f"FROM `{DATABASE}`.`{TARGET_TABLE}` "
        f"WHERE {condition}"
    )

    connection = create_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(discovery_sql)
            matches = cursor.fetchall()
    finally:
        connection.close()

    source_columns: Dict[Tuple[str, str], Set[str]] = {}
    for match in matches:
        source = (str(match["table_name"]), str(match["key_word"]))
        source_columns.setdefault(source, set()).add(str(match["column_name"]))

    result = {
        "query": query,
        "matched_vector_count": len(matches),
    }
    if source_columns:
        result["sql"] = [
            build_source_sql(
                query,
                max_distance,
                table_name,
                key_word,
                sorted(column_names),
            )
            for (table_name, key_word), column_names in sorted(source_columns.items())
        ]
    return result


def recall_many(
    queries: List[str], max_distance: float = DEFAULT_MAX_DISTANCE
) -> List[Dict]:
    unique_queries = dict.fromkeys(query.strip() for query in queries if query.strip())
    return [recall(query, max_distance) for query in unique_queries]
