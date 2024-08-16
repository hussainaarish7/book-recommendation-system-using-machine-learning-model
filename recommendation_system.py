import streamlit as st
import pandas as pd
import pickle

# Load the pickle files
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading file {file_path}: {e}")
        return None

# File paths
popular_books_path = 'top_20_books.pkl'
books_path = 'top_books.pkl'
model_knn_path = 'model_knn.pkl'
ratings_pivot_path = 'ratings_pivot.pkl'

# Load data
popular_books = load_pickle(popular_books_path)
books = load_pickle(books_path)
model_knn = load_pickle(model_knn_path)
final_ratings_pivot = load_pickle(ratings_pivot_path)

# Streamlit app
st.title("Book Recommendation App")

# Function to display book information
def display_book_info(book):
    st.image(book['img_l'], caption=book['book_title'], use_column_width=True)
    st.write(f"**Author:** {book['book_author']}")
    st.write(f"**Votes:** {book['num_ratings']}")
    st.write(f"**Rating:** {book['avg_rating']:.2f}")

# Function to get recommendations
def get_recommendations(book_title):
    try:
        # Check if the book title exists in the books dataframe
        if book_title not in books['book_title'].values:
            st.warning("Book title not found in the database.")
            return pd.DataFrame()

        # Get the ISBN for the input book
        isbn = books[books['book_title'] == book_title]['isbn'].iloc[0]

        # Check if the ISBN exists in the ratings pivot table
        if isbn not in final_ratings_pivot.index:
            st.warning("ISBN not found in the ratings data.")
            return pd.DataFrame()

        # Get the index of the input book in the pivot table
        book_index = final_ratings_pivot.index.get_loc(isbn)

        # Find the k nearest neighbors of the input book
        distances, indices = model_knn.kneighbors(
            final_ratings_pivot.iloc[book_index, :].values.reshape(1, -1), 
            n_neighbors=6
        )

        # Get the ISBNs of the nearest neighbors
        neighbor_isbns = [final_ratings_pivot.index[i] for i in indices.flatten()[1:]]

        # Filter books from the books dataframe
        top_books = books[books['isbn'].isin(neighbor_isbns)]

        # Ensure that the number of neighbors is correctly handled
        if len(neighbor_isbns) != len(top_books):
            st.warning("Not enough similar books found.")

            return pd.DataFrame()

        # Add a column for the distance to the input book
        distances = distances.flatten()[1:]  # Exclude the first element which is the book itself
        top_books['distance'] = distances

        # Drop the images columns
        top_books.drop(['img_s', 'img_m'], axis=1, inplace=True)

        # Sort the books by distance in ascending order
        top_books = top_books.sort_values('distance').head(5)  # Limit to 5 recommendations

        return top_books

    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return pd.DataFrame()  # Return an empty DataFrame

# Initialize session state if not already set
if 'selected_book' not in st.session_state:
    st.session_state.selected_book = None

# Sidebar for navigation
menu = st.sidebar.radio("Navigation", ["Home", "Recommender", "About"])

if menu == "Home":
    # Home Page
    if st.session_state.selected_book is None:
        num_columns = 4
        num_books = len(popular_books)

        # Calculate the number of rows needed
        num_rows = (num_books + num_columns - 1) // num_columns

        # Iterate over rows and columns to display books
        for i in range(num_rows):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                book_index = i * num_columns + j
                if book_index < num_books:
                    book = popular_books.iloc[book_index]
                    with cols[j]:
                        if st.button(f"Detail", key=f"book_{book_index}"):
                            st.session_state.selected_book = book_index
                            st.rerun()
                        st.image(book['img_l'], caption=book['book_title'], use_column_width=True)
    else:
        book = popular_books.iloc[st.session_state.selected_book]
        display_book_info(book)

        if st.button("Go Back"):
            st.session_state.selected_book = None
            st.rerun()

elif menu == "Recommender":
    # Recommender Page
    st.write("### Get Book Recommendations")
    book_title = st.text_input("Enter a book title to get recommendations")

    if st.button('Submit'):
        if book_title:
            recommendations = get_recommendations(book_title)

            if recommendations.empty:
                st.write("No book found with that title or no recommendations available. Please try another book title.")
            else:
                st.write("### Recommended Books")
                num_recommendations = len(recommendations)

                # Create exactly the number of columns needed
                cols = st.columns(5)  # Create 5 columns, but only use the needed ones
                for i in range(num_recommendations):
                    with cols[i]:
                        book = recommendations.iloc[i]
                        st.image(book['img_l'], use_column_width=True)  # Display large image
                        st.write(book['book_title'])
                        st.write(f"**Author:** {book['book_author']}")
                        st.write(f"**Publisher:** {book['publisher']}")
                        st.write(f"**Year:** {book['year_of_publication']:.0f}")

elif menu == "About":
    # About Page
    st.write("### About")
    st.write("""Welcome to the Book Recommendation App! This app is designed to help you explore popular books and discover new reads based on your preferences.

Home Page: Browse through the top 20 most popular books, beautifully displayed in a grid layout. Click on any book to view detailed information, including the cover image, author, number of ratings, and average rating.

Recommender Page: Enter the title of a book you like, and the app will provide personalized recommendations using a K-Nearest Neighbors (KNN) model. Discover similar books that match your taste.

This app combines popularity-based book browsing with an intelligent recommendation engine, making it easier than ever to find your next great read. Enjoy exploring and happy reading!""")

# Optional: Display the dataframe
if menu == "Home":
    st.write("### Detailed View of Popular Books")
    st.dataframe(popular_books)

    # Optional: Provide a summary section
    st.write("### Summary")
    st.write(f"Total Books: {len(popular_books)}")
    average_rating = popular_books['avg_rating'].mean()
    st.write(f"Average Rating Across All Books: {average_rating:.2f}")
