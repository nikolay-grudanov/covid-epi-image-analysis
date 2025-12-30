"""
Модуль для категоризации возраста пациентов с использованием PySpark UDF.

Предоставляет пользовательскую функцию (UDF) для распределения пациентов
по возрастным группам для последующего анализа данных.
"""

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


@udf(StringType())
def categorize_age(age: int) -> str:
    """
    Категоризирует возраст пациента в одну из трёх групп.

    Функция разделяет пациентов на возрастные категории для проведения
    статистического анализа и построения визуализаций.

    Категории:
        - "Junior": пациенты младше 30 лет
        - "Middle": пациенты от 30 до 59 лет
        - "Senior": пациенты 60 лет и старше

    Args:
        age (int): Возраст пациента в годах.

    Returns:
        str: Категория возраста ("Junior", "Middle" или "Senior").

    Examples:
        >>> from pyspark.sql.functions import col
        >>> from pyspark.sql import SparkSession
        >>>
        >>> # Создание тестового DataFrame
        >>> spark = SparkSession.builder.appName("Test").getOrCreate()
        >>> df = spark.createDataFrame([(25,), (45,), (70,)], ["age"])
        >>>
        >>> # Применение UDF для категоризации возраста
        >>> df = df.withColumn("age_group", categorize_age(col("age")))
        >>> df.show()
        +---+---------+
        |age|age_group|
        +---+---------+
        | 25|  Junior|
        | 45|  Middle|
        | 70|  Senior|
        +---+---------+
    """
    if age is None:
        return None

    if age < 30:
        return "Junior"
    elif age < 60:
        return "Middle"
    else:
        return "Senior"


# Пример использования в DataFrame:
# from pyspark.sql.functions import col
#
# # Применение UDF к DataFrame с возрастами пациентов
# df = df.withColumn("age_group", categorize_age(col("age")))
#
# # Проверка результата
# df.select("age", "age_group").show()
