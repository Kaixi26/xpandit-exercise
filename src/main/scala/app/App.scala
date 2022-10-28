package app

import org.apache.commons.lang.SystemUtils
import org.apache.spark.sql
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DateType, DoubleType, LongType}
import org.apache.spark.sql.{SaveMode, SparkSession}

import scala.language.postfixOps
import scala.sys.process._

object App {

  private val ss = SparkSession
    .builder()
    .appName("Spark SQL basic example")
    .master("local[2]")
    .config("spark.sql.parquet.compression.codec", "gzip")
    .getOrCreate()

  import ss.implicits._

  private val unitMap = Map("k" -> 1e3, "M" -> 1e6, "G" -> 1e9)
  val convertUnitsAsDoubleUDF: UserDefinedFunction = udf((s: String) => {
    var r: String = null
    for ((k, v) <- unitMap.iterator.takeWhile(_ => r == null)) {
      if (s.endsWith(k)) {
        r = (s.slice(0, s.lastIndexOf(k)).toDouble * v).toString
      }
    }
    r
  })

  def part1(userReviews: sql.DataFrame): sql.DataFrame = {

    // Select relevant columns
    var df = userReviews.select($"App", $"Sentiment_Polarity")

    // Cast 'Sentiment_Polarity' to double
    df = df.withColumn("Sentiment_Polarity", $"Sentiment_Polarity".cast(DoubleType))

    // Group by the column App while aggregating 'Sentiment_Polarity by the average
    df = df.groupBy($"App")
      .agg(avg("Sentiment_Polarity").as("Average_Sentiment_Polarity"))

    // Make 0 the default value
    df = df.na.fill(0, Seq("Average_Sentiment_Polarity"))

    df
  }

  def part2(googlePlayStore: sql.DataFrame): sql.DataFrame = {

    // Select relevant columns
    var df = googlePlayStore.withColumn("Rating", $"Rating".cast(DoubleType))

    // Filter Apps with ratings greater than 4 and sort them in descending Rating order
    df = df.filter($"Rating" >= 4 && !isnan($"Rating"))
      .sort($"Rating".desc)

    df.repartition(1) // Repartition to generate only one file
      .write
      .option("header", value = true)
      .option("delimiter", "§") // Change delimiter to §
      .mode(SaveMode.Overwrite)
      .csv("best_apps_csv")

    // Copy csv generated into the folder to the file 'best_apps.csv'
    //TODO: implement copy directly in java instead of executing unix command
    if (SystemUtils.IS_OS_UNIX) {
      Seq("bash", "-c", "cp best_apps_csv/*.csv best_apps.csv") !
    } else {
      System.err.println("System is not unix, cannot use bash to copy file.")
    }

    df
  }

  def part3(googlePlayStore: sql.DataFrame): sql.DataFrame = {
    var gpstore: sql.DataFrame = googlePlayStore

    // Rename columns
    Map("Content Rating" -> "Content_Rating", "Last Updated" -> "Last_Updated", "Current Ver" -> "Current_Version", "Android Ver" -> "Minimum_Android_Version")
      .foreach(m => gpstore = gpstore.withColumnRenamed(m._1, m._2))

    // Update columns to expected types
    Map("Rating" -> DoubleType, "Reviews" -> LongType, "Price" -> DoubleType)
      .foreach(m => gpstore = gpstore.withColumn(m._1, gpstore(m._1).cast(m._2)))

    // Perform the more specific type transformations
    gpstore = gpstore.select($"*"
      , split($"Genres", ";").as("_Genres")
      , to_date($"Last_updated", "MMMM dd, yyyy").as("_Last_updated")
      , convertUnitsAsDoubleUDF($"Size").as("_Size").cast(DoubleType)
      , ($"Price" * 0.9).as("_Price"))

    // Swap new columns to original position
    gpstore = gpstore.select(gpstore.columns.flatMap {
      case "Genres" => Some(gpstore("_Genres").as("Genres"))
      case "Size" => Some(gpstore("_Size").as("Size"))
      case "Last_Updated" => Some(gpstore("_Last_Updated").as("Last_Updated"))
      case "Price" => Some(gpstore("_Price").as("Price"))
      case colName => if (colName.startsWith("_")) None else Some(gpstore(colName))
    }: _*)

    // Make 0 the default value for reviews
    gpstore.na.fill(Map("Reviews" -> 0))

    var df: sql.DataFrame = gpstore

    // Group by app and collect categories as set (removes duplicates) and the max value for Reviews
    df = df.groupBy($"App")
      .agg(collect_set($"Category").alias("Categories")
        , max($"Reviews").alias("Reviews")
      )

    // Join with original table based on previously calculated max number of reviews
    // Left join assures only one row is selected in case multiple columns have the same max number of reviews
    gpstore = gpstore
      .withColumnRenamed("App", "_App")
      .withColumnRenamed("Reviews", "_Reviews")
    df = df.join(gpstore, df("App") === gpstore("_App") && df("Reviews") === gpstore("_Reviews"), "left")
      .drop("_App", "_Reviews", "Category")
      .dropDuplicates("App", "Reviews")

    // Swap Reviews and Rating to original position
    df = df.select(df.columns.flatMap {
      case "Reviews" => Some(df("Rating"))
      case "Rating" => Some(df("Reviews"))
      case colName => if (colName.startsWith("_")) None else Some(df(colName))
    }: _*)

    df
  }

  def part4(df_3: sql.DataFrame, df_1: sql.DataFrame): sql.DataFrame = {
    val df_1r = df_1.withColumnRenamed("App", "_App")
    var df = df_3

    // Join columns for same App
    df = df.join(df_1r, df("App") === df_1r("_App"), "left").drop("_App")

    df = df.na.fill("null", Array("Average_Sentiment_Polarity"))

    // Write as parquet
    // gzip compression was already set before, in the SparkSession config
    df.write
      .option("header", value = true)
      .mode(SaveMode.Overwrite)
      .parquet("googleplaystore_cleaned")

    df
  }

  def part5(df_4: sql.DataFrame): sql.DataFrame = {
    var df = df_4

    // Select relevant columns
    // Total sentiment polarity is calculated based on average an total reviews,
    //   this means when later grouping by Genre we can recalculate the average
    //   sentiment polarity for that Genre
    df = df.select(
      explode($"Genres").as("Genre")
      , $"App"
      , $"Rating".cast(DoubleType)
      , ($"Average_Sentiment_Polarity" * $"Reviews").as("Total_Sentiment_Polarity")
      , $"Reviews"
    )
    // cleanup NaNs so they dont affect averages
    df = df.na.fill(Map("Rating" -> "null"))

    // Group by Genre and calculate total amount of apps for that Genre,
    // average ratings, and the average sentiment polarity
    df = df.groupBy($"Genre")
      .agg(count($"App").as("Count")
        , avg($"Rating").as("Average_Rating")
        , (sum($"Total_Sentiment_Polarity") / sum($"Reviews")).as("Average_Sentiment_Polarity")
      )

    // Write as parquet
    // gzip compression was already set before, in the SparkSession config
    df.write
      .option("header", value = true)
      .mode(SaveMode.Overwrite)
      .parquet("googleplaystore_metrics")

    df
  }

  def main(args: Array[String]): Unit = {
    ss.sparkContext.setLogLevel("WARN")

    println("Reading files...")

    val googlePlayStore: sql.DataFrame = ss.read
      .option("header", value = true)
      .csv("data/googleplaystore.csv")

    val userReviews: sql.DataFrame = ss.read
      .option("header", value = true)
      .csv("data/googleplaystore_user_reviews.csv")

    println("Executing part 1...")
    val df_1 = part1(userReviews)
    df_1.printSchema()
    df_1.show(5)

    println("Executing part 2...")
    val df_2 = part2(googlePlayStore)
    df_2.printSchema()
    df_2.show(5)

    println("Executing part 3...")
    val df_3 = part3(googlePlayStore)
    df_3.printSchema()
    df_3.show(5)

    println("Executing part 4...")
    val df_4 = part4(df_3, df_1)
    df_4.printSchema()
    df_4.filter(!isnull($"Average_Sentiment_Polarity")).show(5)

    println("Executing part 5...")
    val df_5 = part5(df_4)
    df_5.printSchema()
    df_5.show(5)

    println("All done :)")

    ss.stop()
  }
}
