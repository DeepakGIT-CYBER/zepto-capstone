import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"

# Fixed conversion rate required for this project
GBP_TO_INR = 105.50

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "books.db"
OUTPUT_PATH = BASE_DIR / "query_outputs.txt"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# ============================================================
# 1. SCRAPE DATA
# ============================================================

def scrape_books():
    books = []

    print("=" * 70)
    print("STARTING BOOK SCRAPER")
    print("=" * 70)

    # Scrape first 5 pages = 100 books
    for page in range(1, 6):

        url = BASE_URL.format(page)

        print(f"\nScraping page {page}: {url}")

        response = requests.get(
            url,
            timeout=20,
            headers=HEADERS
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        articles = soup.select("article.product_pod")

        print(f"Books found on page {page}: {len(articles)}")

        for article in articles:

            # ------------------------------------------------
            # Basic book information
            # ------------------------------------------------

            title_tag = article.select_one("h3 a")
            price_tag = article.select_one(".price_color")
            rating_tag = article.select_one(".star-rating")
            availability_tag = article.select_one(".availability")

            title = (
                title_tag.get("title", "").strip()
                if title_tag
                else "Unknown"
            )

            price = (
                price_tag.get_text(strip=True)
                if price_tag
                else ""
            )

            # ------------------------------------------------
            # Rating
            # ------------------------------------------------

            if rating_tag:
                rating_classes = rating_tag.get("class", [])

                if len(rating_classes) >= 2:
                    star_rating = rating_classes[1]
                else:
                    star_rating = ""
            else:
                star_rating = ""

            # ------------------------------------------------
            # Availability
            # ------------------------------------------------

            availability = (
                availability_tag.get_text(
                    " ",
                    strip=True
                )
                if availability_tag
                else ""
            )

            # ------------------------------------------------
            # Category
            # ------------------------------------------------

            category = "Unknown"

            product_link = (
                title_tag.get("href")
                if title_tag
                else None
            )

            if product_link:

                product_url = requests.compat.urljoin(
                    url,
                    product_link
                )

                try:

                    product_response = requests.get(
                        product_url,
                        timeout=20,
                        headers=HEADERS
                    )

                    product_response.raise_for_status()

                    product_soup = BeautifulSoup(
                        product_response.text,
                        "html.parser"
                    )

                    breadcrumb = product_soup.select(
                        "ul.breadcrumb li a"
                    )

                    if len(breadcrumb) >= 3:

                        category = breadcrumb[2].get_text(
                            strip=True
                        )

                except requests.RequestException:

                    category = "Unknown"

            # ------------------------------------------------
            # Store book
            # ------------------------------------------------

            books.append(
                {
                    "title": title,
                    "price": price,
                    "star_rating": star_rating,
                    "availability": availability,
                    "category": category
                }
            )

    df = pd.DataFrame(books)

    print("\nScraping completed.")
    print(f"Total books scraped: {len(df)}")

    return df


# ============================================================
# 2. CLEAN DATA
# ============================================================

def clean_data(df):

    print("\n" + "=" * 70)
    print("CLEANING DATA")
    print("=" * 70)

    # Make a copy so original scraped dataframe is preserved
    df = df.copy()

    # --------------------------------------------------------
    # Price: extract numeric GBP value
    # --------------------------------------------------------

    df["price_gbp"] = (
        df["price"]
        .astype(str)
        .str.extract(
            r"(\d+(?:\.\d+)?)",
            expand=False
        )
    )

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Handle missing GBP prices
    # --------------------------------------------------------

    missing_prices = df["price_gbp"].isna().sum()

    if missing_prices > 0:

        print(
            f"Missing GBP prices found: {missing_prices}"
        )

        median_price = df["price_gbp"].median()

        if pd.isna(median_price):

            raise ValueError(
                "Could not parse any GBP prices."
            )

        df["price_gbp"] = df["price_gbp"].fillna(
            median_price
        )

    # --------------------------------------------------------
    # Rating text -> integer
    # --------------------------------------------------------

    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    df["rating"] = df["star_rating"].map(
        rating_map
    )

    # --------------------------------------------------------
    # Handle missing ratings
    # --------------------------------------------------------

    if df["rating"].isna().any():

        median_rating = df["rating"].median()

        if pd.isna(median_rating):
            median_rating = 3

        median_rating = int(
            round(median_rating)
        )

        df["rating"] = df["rating"].fillna(
            median_rating
        )

    df["rating"] = df["rating"].astype(int)

    # --------------------------------------------------------
    # Availability -> boolean
    # --------------------------------------------------------

    df["in_stock"] = (
        df["availability"]
        .astype(str)
        .str.lower()
        .str.contains(
            "in stock",
            na=False
        )
    )

    # --------------------------------------------------------
    # GBP -> INR
    # --------------------------------------------------------

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    )

    # --------------------------------------------------------
    # Clean category
    # --------------------------------------------------------

    df["category"] = (
        df["category"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    df.loc[
        df["category"] == "",
        "category"
    ] = "Unknown"

    # --------------------------------------------------------
    # Keep required columns
    # --------------------------------------------------------

    df = df[
        [
            "title",
            "price_gbp",
            "rating",
            "in_stock",
            "price_inr",
            "category"
        ]
    ]

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if df["price_gbp"].isna().any():
        raise ValueError(
            "price_gbp still contains NULL values."
        )

    if df["price_inr"].isna().any():
        raise ValueError(
            "price_inr still contains NULL values."
        )

    if df["rating"].isna().any():
        raise ValueError(
            "rating still contains NULL values."
        )

    if df["category"].isna().any():
        raise ValueError(
            "category still contains NULL values."
        )

    print(f"Cleaned rows: {len(df)}")

    print("\nCleaned columns:")
    print(df.columns.tolist())

    print("\nSample cleaned data:")
    print(
        df.head().to_string(
            index=False
        )
    )

    return df


# ============================================================
# 3. CREATE SQLITE DATABASE
# ============================================================

def create_database(df):

    print("\n" + "=" * 70)
    print("CREATING SQLITE DATABASE")
    print("=" * 70)

    # Remove old database if it exists
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(
        DB_PATH
    )

    cursor = conn.cursor()

    cursor.execute(
        "PRAGMA foreign_keys = ON"
    )

    # --------------------------------------------------------
    # Categories table
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
        """
    )

    # --------------------------------------------------------
    # Books table
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,

            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
        """
    )

    # --------------------------------------------------------
    # Insert categories
    # --------------------------------------------------------

    categories = sorted(
        df["category"]
        .dropna()
        .unique()
        .tolist()
    )

    for category in categories:

        cursor.execute(
            """
            INSERT INTO categories
            (category_name)
            VALUES (?)
            """,
            (category,)
        )

    # --------------------------------------------------------
    # Create category -> ID mapping
    # --------------------------------------------------------

    rows = cursor.execute(
        """
        SELECT category_id, category_name
        FROM categories
        """
    ).fetchall()

    category_map = {
        category_name: category_id
        for category_id, category_name in rows
    }

    # --------------------------------------------------------
    # Insert books
    # --------------------------------------------------------

    for _, row in df.iterrows():

        category_id = category_map[
            row["category"]
        ]

        cursor.execute(
            """
            INSERT INTO books
            (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(row["title"]),
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(row["in_stock"]),
                int(category_id)
            )
        )

    conn.commit()

    # --------------------------------------------------------
    # Validate database
    # --------------------------------------------------------

    book_count = cursor.execute(
        "SELECT COUNT(*) FROM books"
    ).fetchone()[0]

    category_count = cursor.execute(
        "SELECT COUNT(*) FROM categories"
    ).fetchone()[0]

    print(f"Database created: {DB_PATH}")
    print(f"Books stored: {book_count}")
    print(f"Categories stored: {category_count}")

    if book_count == 0:
        conn.close()
        raise RuntimeError(
            "Database contains zero books."
        )

    return conn


# ============================================================
# 4. SQL QUERIES + PANDAS
# ============================================================

def run_queries(conn):

    print("\n" + "=" * 70)
    print("RUNNING SQL QUERIES")
    print("=" * 70)

    queries = {

        # ----------------------------------------------------
        # Q1 - SELECT + WHERE
        # ----------------------------------------------------

        "Q1_SELECT_WHERE": """
            SELECT
                title,
                price_gbp,
                rating
            FROM books
            WHERE rating >= 4
        """,

        # ----------------------------------------------------
        # Q2 - ORDER BY
        # ----------------------------------------------------

        "Q2_ORDER_BY": """
            SELECT
                title,
                price_gbp
            FROM books
            ORDER BY price_gbp DESC
            LIMIT 10
        """,

        # ----------------------------------------------------
        # Q3 - DISTINCT
        # ----------------------------------------------------

        "Q3_DISTINCT": """
            SELECT DISTINCT
                category_name
            FROM categories
            ORDER BY category_name
        """,

        # ----------------------------------------------------
        # Q4 - BETWEEN
        # ----------------------------------------------------

        "Q4_BETWEEN": """
            SELECT
                title,
                price_gbp,
                rating
            FROM books
            WHERE price_gbp BETWEEN 20 AND 40
        """,

        # ----------------------------------------------------
        # Q5 - IN
        # ----------------------------------------------------

        "Q5_IN": """
            SELECT
                title,
                rating,
                in_stock
            FROM books
            WHERE rating IN (4, 5)
            LIMIT 15
        """,

        # ----------------------------------------------------
        # Q6 - JOIN
        # ----------------------------------------------------

        "Q6_JOIN": """
            SELECT
                b.title,
                c.category_name,
                b.price_gbp,
                b.rating,
                b.in_stock
            FROM books b
            JOIN categories c
                ON b.category_id = c.category_id
            ORDER BY
                b.rating DESC,
                b.price_gbp DESC
            LIMIT 10
        """
    }

    output_lines = []

    output_lines.append(
        "ZEPTO DATA PIPELINE - SQL QUERY OUTPUTS"
    )

    output_lines.append(
        "=" * 70
    )

    output_lines.append(
        f"Fixed conversion rate: "
        f"1 GBP = {GBP_TO_INR:.2f} INR"
    )

    output_lines.append("")

    sql_results = {}

    # --------------------------------------------------------
    # Execute all SQL queries
    # --------------------------------------------------------

    for name, query in queries.items():

        print("\n" + name)
        print("-" * 70)

        result = pd.read_sql_query(
            query,
            conn
        )

        sql_results[name] = result

        print(
            result.to_string(
                index=False
            )
        )

        output_lines.append(name)
        output_lines.append("SQL:")
        output_lines.append(query.strip())
        output_lines.append("")
        output_lines.append("OUTPUT:")
        output_lines.append(
            result.to_string(
                index=False
            )
        )
        output_lines.append("")
        output_lines.append(
            "=" * 70
        )

    # --------------------------------------------------------
    # Q1 through pandas.read_sql
    # --------------------------------------------------------

    q1_df = pd.read_sql_query(
        queries["Q1_SELECT_WHERE"],
        conn
    )

    print("\nPANDAS pd.read_sql RESULT")
    print(
        q1_df.head().to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SQL JOIN result
    # --------------------------------------------------------

    q6_sql_df = pd.read_sql_query(
        queries["Q6_JOIN"],
        conn
    )

    # --------------------------------------------------------
    # Reproduce JOIN using pandas.merge
    # --------------------------------------------------------

    category_df = pd.read_sql_query(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        conn
    )

    books_df = pd.read_sql_query(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            rating,
            in_stock,
            category_id
        FROM books
        """,
        conn
    )

    merged_df = pd.merge(
        books_df,
        category_df,
        on="category_id",
        how="inner"
    )

    merged_df = (
        merged_df[
            [
                "title",
                "category_name",
                "price_gbp",
                "rating",
                "in_stock"
            ]
        ]
        .sort_values(
            ["rating", "price_gbp"],
            ascending=[False, False]
        )
        .head(10)
        .reset_index(drop=True)
    )

    q6_sql_df = q6_sql_df.reset_index(
        drop=True
    )

    print("\nSQL JOIN RESULT:")
    print(
        q6_sql_df.to_string(
            index=False
        )
    )

    print("\nPANDAS MERGE RESULT:")
    print(
        merged_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Compare SQL JOIN and pandas.merge
    # --------------------------------------------------------

    equivalent = q6_sql_df.equals(
        merged_df
    )

    print(
        "\nSQL JOIN vs pandas.merge equivalent:",
        equivalent
    )

    output_lines.append("")
    output_lines.append(
        "SQL JOIN vs pandas.merge"
    )

    output_lines.append(
        "SQL JOIN RESULT:"
    )

    output_lines.append(
        q6_sql_df.to_string(
            index=False
        )
    )

    output_lines.append("")

    output_lines.append(
        "PANDAS MERGE RESULT:"
    )

    output_lines.append(
        merged_df.to_string(
            index=False
        )
    )

    output_lines.append("")

    output_lines.append(
        f"Equivalent output: {equivalent}"
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    OUTPUT_PATH.write_text(
        "\n".join(output_lines),
        encoding="utf-8"
    )

    print(
        f"\nQuery outputs saved to: "
        f"{OUTPUT_PATH}"
    )

    return sql_results


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # Scrape
        # ----------------------------------------------------

        raw_df = scrape_books()

        print(
            "\nRaw dataset shape:",
            raw_df.shape
        )

        # ----------------------------------------------------
        # Validate scraped data
        # ----------------------------------------------------

        if len(raw_df) < 60:

            raise RuntimeError(
                f"Only {len(raw_df)} books scraped. "
                "At least 60 are required."
            )

        category_count = (
            raw_df["category"]
            .nunique()
        )

        if category_count < 3:

            raise RuntimeError(
                "Fewer than 3 categories were found."
            )

        # ----------------------------------------------------
        # Clean
        # ----------------------------------------------------

        clean_df = clean_data(
            raw_df
        )

        # ----------------------------------------------------
        # Show final data
        # ----------------------------------------------------

        print("\nFINAL DATASET")
        print("-" * 70)

        print(
            clean_df.head(10).to_string(
                index=False
            )
        )

        print("\nDATA TYPES")
        print("-" * 70)

        print(
            clean_df.dtypes
        )

        # ----------------------------------------------------
        # Create database
        # ----------------------------------------------------

        conn = create_database(
            clean_df
        )

        # ----------------------------------------------------
        # Run SQL queries
        # ----------------------------------------------------

        run_queries(
            conn
        )

        # ----------------------------------------------------
        # Close database
        # ----------------------------------------------------

        conn.close()

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(
            f"\nDatabase: {DB_PATH}"
        )

        print(
            f"Query output: {OUTPUT_PATH}"
        )

    except requests.RequestException as error:

        print(
            "\nNETWORK ERROR:"
        )

        print(error)

    except sqlite3.Error as error:

        print(
            "\nDATABASE ERROR:"
        )

        print(error)

    except Exception as error:

        print(
            "\nPIPELINE ERROR:"
        )

        print(error)

        raise


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()