import pickle
import random
import wikipedia
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer


def get_input(user_name):  # Gets input from user, repeats until user writes something
    user_input = ''
    while not user_input:
        user_input = input(user_name + ': ')
    # If the user wrote bye, it will signify the end of the conversation
    if 'bye' in user_input.lower():
        return ''
    return user_input


def search_game(input):  # Parses user input and searches on Wikipedia. Outputs information about the game if successful
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    test_list = ['like', 'love', 'favorite', 'really', 'best', 'video', 'game', 'hate', 'don\'t']  # Try to remove unnecessary words
    tokens = word_tokenize(input)
    tokens = [t for t in tokens if t not in stopwords.words('english')]  # Not in stopwords
    tokens = [t for t in tokens if t not in test_list]
    input = (' ').join(tokens)
    # Search Wikipedia. Suggestion means it will also return what it thought the user meant
    # Returns in the form of a tuple. The first term is the list of results, the second is the string it is suggesting
    results = wikipedia.search(input, results=3, suggestion=True)
    if results[1]:  # If there was a suggestion, add it to the list
        results = results[0] + wikipedia.search(results[1], results=3)
    else:
        results = results[0]
    if results:  # If there was a result
        for game in results:  # Look through each item and determine if the page is the page for a game
            try:
                page = wikipedia.page(game, auto_suggest=False)
                if 'game' in page.content:
                    # Get the table that is on the left side of a typical Wikipedia page
                    df = pandas.read_html(page.html().encode('UTF-8'))
                    df = df[0]
                    # Get the genres and the developer names
                    i = df.index[df[game] == 'Genre(s)']
                    genres = df.iloc[i[0], 1].lower()
                    i = df.index[df[game] == 'Developer(s)']
                    developers = df.iloc[i[0], 1]
                    print('CHATBOT: ' + game + ' is a ' + genres + ' game developed by ' + developers[:25] + '.')
                    return game  # Return the name of the game
            except:
                pass
    return ''


def process_input(raw_text):  # Tokenize and perform basic preprocessing
  from nltk.tokenize import word_tokenize
  from nltk.corpus import stopwords
  tokens = word_tokenize(raw_text)
  tokens = [t.lower() for t in tokens]  # Make everything lower case
  tokens = [t for t in tokens if t.isalpha()]  # Only include tokens that only have letters
  tokens = [t for t in tokens if t not in stopwords.words('english')]  # Not in stopwords
  tokens = [t for t in tokens if len(t) > 1]  # Length > 1
  return tokens


def yes_or_no(input):  # Tries to determine if the user confirmed or denied
    input = process_input(input)

    list_yes = {'y', 'ye', 'yes', 'yea', 'yeah', 'yep', 'sure', 'right', 'like', 'love', 'ok', 'fun', 'good', 'great'}
    if bool(set(input) & list_yes):
        return True
    else:
        return False


def pos_tag(input):  # Performs PoS tagging; keeps certain tags
    input = process_input(input)
    test_list = ['game', 'games', 'video']  # Try to remove unnecessary words
    input = [t for t in input if t not in test_list]
    terms  = nltk.pos_tag(input)
    terms = [t for t, pos in terms if pos in ['NN', 'NNS', 'JJ', 'VBG']]
    return terms


def build_tfidf(tfidf_vectorizer):  # Used to build tf-idf matrix. Pickles the variables for later use.
    # Wikipedia's page on the list of best video games is used. The program will recommend a game in this list.

    # Get the column with the video game names
    site = wikipedia.search('List of video games considered the best')[0]
    page = wikipedia.page(site, auto_suggest=False)
    df = pandas.read_html(page.html().encode('UTF-8'))
    list_games = df[1]['Game'].tolist()

    # Search for the game in Wikipedia and get text. If it fails, remove it from the list of games
    list_new = []
    docs = []
    for game in list_games:
        site = wikipedia.search(game)[0]
        try:
            page = wikipedia.page(site, auto_suggest=False)
            text = page.content.split('==')
            raw_text = text[0]
            tokens = process_input(raw_text)
            docs.append(' '.join(tokens))
            list_new.append(game)
        except:
            pass

    # Use SKLearn's tfidf vectorizer
    tfidf_docs = tfidf_vectorizer.fit_transform(docs)

    # Get feature names for later use
    words = tfidf_vectorizer.get_feature_names_out()

    # Save as pickles
    pickle.dump(tfidf_docs, open('tfidf.p', 'wb'), protocol=4)
    pickle.dump(words, open('words.p', 'wb'))
    pickle.dump(list_new, open('list_games.p', 'wb'))


def recommend(query, list_games, words):  # Recommend a game given some information about what the user likes
    # Use the tf-idf matrix to get the cosine similarities. Return the index with the highest one
    queryTFIDF = TfidfVectorizer().fit(words)
    queryTFIDF = queryTFIDF.transform([query])
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_similarities = cosine_similarity(queryTFIDF, tfidf_docs).flatten()

    # Get max index
    max = 0
    index = 0
    for i, x in enumerate(cosine_similarities):
        if x > max:
            max = x
            index = i

    return index

    doc = 247
    feature_index = tfidf_docs[doc, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_docs[doc, x] for x in feature_index])
    for w, s in [(words[i], s) for (i, s) in tfidf_scores]:
        print(w, s)


if __name__ == '__main__':
    random.seed()
    #tfidf_vectorizer = TfidfVectorizer()
    #build_tfidf(tfidf_vectorizer)  # Used to build tf-idf matrix. Takes a very long time. The pickles have already been provided.
    # Read pickles
    tfidf_docs = pickle.load(open('tfidf.p', 'rb'))
    words = pickle.load(open('words.p', 'rb'))
    list_games = pickle.load(open('list_games.p', 'rb'))
    
    liked_games = []  # list of games the user liked
    disliked_games = []
    liked_qualities = []  # list of characteristics user likes
    disliked_qualities = []

    try:  # Find if user data exists
        f = open('user.txt', 'r')  # Parse data
        data = f.read()
        f.close()
        data = data.split('\n')
        # Read user data
        user_name = data[0]
        if data[1]:
            liked_games = data[1].split(',')  # list of games the user liked
        if data[2]:
            disliked_games = data[2].split(',')
        if data[3]:
            liked_qualities = data[3].split(',')   # list of characteristics user likes
        if data[4]:
            disliked_qualities = data[4].split(',')
        print('CHATBOT: Welcome back, ' + user_name + '.')

    except:  # Create new user data
        print('CHATBOT: Hello. What is your name?')
        user_name = input('You: ')
        print('CHATBOT: Nice to meet you, ' + user_name + '. I\'m here to talk about video games.')

    user_input = '-'
    previous = -1  # Don't repeat the same topic
    while user_input:  # Conversation loop. Ends if user says bye
        random_int = random.randint(0, 7)  # Randomly decides the next topic
        if random_int == previous:
            continue
        if random_int == 0:
            print('CHATBOT: What is a video game you like?')
            user_input = get_input(user_name)
            if not user_input:
                break
            game = search_game(user_input)
            if game:
                liked_games.append(game)
            else:
                print('CHATBOT: What do you like about it?')
                user_input = get_input(user_name)
                if not user_input:
                    break
                terms = pos_tag(user_input)
                if terms:
                    liked_qualities += terms
                    print('CHATBOT: You like it for the ' + random.choice(terms) + '? I see.')
                else:
                    print('CHATBOT: I see.')
        elif random_int == 1:
            game = random.choice(list_games)
            print('CHATBOT: I really like the game ' + game + '. What do you think about it?')
            user_input = get_input(user_name)
            if not user_input:
                break
            if yes_or_no(user_input):
                print('CHATBOT: Glad to hear it.')
                liked_games.append(game)
            else:
                print('CHATBOT: Oh.')
                disliked_games.append(game)
        elif random_int == 2 and liked_qualities:
            print('CHATBOT: I can recommend you a game I think you would like.')
            index = recommend((' ').join(liked_qualities), list_games, words)
            game = list_games[index]
            print('CHATBOT: If you like ' + random.choice(liked_qualities) + ', I recommend ' + game + '. What do you think about that one?')
            user_input = get_input(user_name)
            if not user_input:
                break
            if yes_or_no(user_input):
                print('CHATBOT: Good!')
                liked_games.append(game)
            else:
                print('CHATBOT: I\'ll try to find something better.')
                disliked_games.append(game)
        elif random_int == 3 and liked_games:
            temp = list(set(liked_games)-set(list_games))
            if temp:
                game = random.choice(temp)
                print('CHATBOT: I can recommend you a game similar to ' + game + '.')
                page = wikipedia.page(game, auto_suggest=False)
                tokens = process_input(page.summary)
                index = recommend((' ').join(tokens), list_games, words)
                game = list_games[index]
                print('CHATBOT: How about ' + game + '?')
                user_input = get_input(user_name)
                if not user_input:
                    break
                if yes_or_no(user_input):
                    print('CHATBOT: Good!')
                    liked_games.append(game)
                else:
                    print('CHATBOT: I\'ll try to find something better.')
                    disliked_games.append(game)
        elif random_int == 4:
            print('CHATBOT: What is most important to you in a video game?')
            user_input = get_input(user_name)
            if not user_input:
                break
            terms = pos_tag(user_input)
            if terms:
                liked_qualities += terms
                print('CHATBOT: The ' + random.choice(terms) + '? I see.')
            else:
                print('CHATBOT: I see.')
        elif random_int == 5 and liked_games:
            game = random.choice(liked_games)
            print('CHATBOT: What do you like about ' + game + '?')
            user_input = get_input(user_name)
            if not user_input:
                break
            terms = pos_tag(user_input)
            if terms:
                liked_qualities += terms
                print('CHATBOT: You like it for the ' + random.choice(terms) + '? I see.')
            else:
                print('CHATBOT: I see.')
        elif random_int == 6:
            print('CHATBOT: What is a game you dislike?')
            user_input = get_input(user_name)
            if not user_input:
                break
            game = search_game(user_input)
            if game:
                disliked_games.append(game)
                print('CHATBOT: Why don\'t you like it?')
                user_input = get_input(user_name)
                if not user_input:
                    break
                terms = pos_tag(user_input)
                disliked_qualities += terms
            else:
                print('CHATBOT: I see.')
        elif random_int == 7:
            print('CHATBOT: What is something you don\'t like to see in games?')
            user_input = get_input(user_name)
            if not user_input:
                break
            terms = pos_tag(user_input)
            if terms:
                disliked_qualities += terms
                print('CHATBOT: The ' + random.choice(terms) + '? I\'ll try to avoid them.')
            else:
                print('CHATBOT: I see.')

    print('CHATBOT: I hope I can talk to you again.')

    # Write to file
    f = open('user.txt', 'w')
    f.write(user_name + '\n')
    f.write((',').join(liked_games) + '\n')
    f.write((',').join(disliked_games) + '\n')
    f.write((',').join(liked_qualities) + '\n')
    f.write((',').join(disliked_qualities) + '\n')
    f.close()

