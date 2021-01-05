# WikiArt API is very poor. Pagination doesn't work and most of links are broken.
import requests

# TODO: code can be rewritten using API 2, but it has same endopoints + Token Auth will be needed

# Get all Artists by specific Genre
# genre = 'abstract' # seo_name in API
genre = 'flower-painting'  # seo_name in API
page = 1
page_size = 600
url_get_artists_by_genre = 'http://www.wikiart.org/en/Artists-by-Genre/{genre}/{page}?json=1&pagesize={page_size}' \
    .format(genre=genre, page=page, page_size=page_size)
# approxim 500 abstract genre authors
artists = requests.get(url_get_artists_by_genre).json()
# artists = artists[::-1]

count = 0
for artist in artists:
    # Loop through Artists and get all Paintings by that Artist
    artist_url = artist['url']
    url_get_paintings_by_artist = 'https://www.wikiart.org/en/App/Painting/PaintingsByArtist?artistUrl={artist_url}&json=1' \
        .format(artist_url=artist_url)
    paintings = requests.get(url_get_paintings_by_artist).json()
    for painting in paintings:
        # Get full information about painting,
        # since not all paintings from specific genre artist are same genre.
        content_id = painting['contentId']
        url_get_painting_info = 'https://www.wikiart.org/en/App/Painting/ImageJson/{content_id}' \
            .format(content_id=content_id)
        painting_info = requests.get(url_get_painting_info).json()
        # check genre
        if painting_info is None or painting_info['genre'] is None:
            continue
        if len(painting_info['genre']) != len(genre):
            continue
        # Save image in FS
        painting_url = painting_info['image']
        painting_name = painting_info['artistUrl'] + '--' + str(content_id) + '.jpg'

        response = requests.get(painting_url)
        if response.status_code != 200:
            continue

        painting_bytes = response.content
        with open('./dataset/' + painting_name, 'wb') as handler:
            handler.write(painting_bytes)
        count += 1
        print(count)
        if count == 2000:
            exit(1)

# TODO: add photo transformations

# style = 'baroque'
# json = 1 indicates that return type is JSON
# url_get_paintings_by_style = 'https://www.wikiart.org/en/search/Any/1?style={style}&json=2&PageSize=600'
# url_get_paintings_by_style = url_get_paintings_by_style.format(style=style)
# requests.get(url_get_paintings_by_style).json()
