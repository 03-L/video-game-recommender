# video-game-recommender
Rules-based chatbot that asks user about games they like and recommends games based on responses

## Libraries Used
wikipedia,
nltk,
pandas,
scikit-learn / sklearn,
lxml

## Instructions
Install libraries: 
```
pip install [library name]
```
Run: 
```
python main.py
```
End conversation by writing "bye."
## Project Report
### NLP Techniques Used
- Text Processing (process_input):
I performed basic text processing by tokenizing the text and cutting out unimportant words, such as stop words or words that are too short. It is used to process user input and text I get from Wikipedia.

- Parts of Speech Tagging:
When the bot asks the user an open-ended question, such as asking what genres of games the user likes, after preprocessing, I use PoS tagging to get only the nouns or adjectives from what the user writes. I did this to get information that the program can use when it uses the cosine similarity function to recommend a game.

- tf-idf:
Wikipedia has a page called “List of video games considered the best” that lists well-regarded games throughout video game history. I created a tf-idf matrix using just the summary portion of each game in that list. I used the Pandas library to get the list of the games in the page, then used the Wikipedia library to retrieve the text on each page. After I got all the pieces of text, I used SKLearn to create the vectors.

- Cosine Similarity:
With the td-idf matrix and information from the user about their preferences, I used cosine similarity to find the game that matched the most with what the user liked. I used SKLearn for this as well. The chatbot recommends the game with the highest cosine similarity.

The tf-idf matrix is created beforehand (saved as a pickle) using the methods described above.

### Wikipedia Library
I used the Wikipedia library to look up information on Wikipedia in real time. If a user mentions the name of a game, and a Wikipedia page for it exists, the program gets the genre and developer of the game and prints it out accordingly.

### Personal Evaluation
The chatbot is repetitive. The program is able to look up information about a video game in real time, but it is limited to Wikipedia, where some less popular games don’t have pages created. It would be nice to be able to have a long conversation regarding one game, where it talks about the specifics. As of now, the chatbot focuses more on asking the user questions and using the answers to give the user a recommendation.
****
