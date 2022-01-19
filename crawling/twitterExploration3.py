import tweepy as tw
import pandas as pd
import datetime
import os
import emoji
import unicodedata

consumer_key = 'AD0wbHFLUz2QY1eV5keMDdXft'
consumer_secret = '3xdE6riwBOkkzHd9EyqGbiHWfqZc0bUcuSajQ9RI7WuZa9zBrn'
access_token = '1362012670304673796-xptRPoZ7EI9B2Pe4obC8KwQc6EVpSr'
access_token_secret = 'fdoxBfB4IM2yxwCuwxrP7wZxAaLwJwqEC8JCNCtf2fmo1'
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tw.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")

#fix the tweet --> replace emojis, remove @...., remove urls
def fix_text(text):
    emtext = emoji.demojize(text, delimiters=("", ""))
    ret = str(unicodedata.normalize('NFKD', emtext).encode('ascii', 'ignore'))[2:-1]
    ret = ret.replace('\\n'," ")
    words = ret.split()
    keep = []
    for word in words:
        if word[0] == "@":
            continue
        if word[:5] == "https":
            continue
        if word[0] == "#":
            word = word[1:]
        if word[0] == "\"" or word[-1] == "'":
            word = word[1:]
        if word[-1] == "\"" or word[-1] == "'":
            word = word[:-1]
        ls = word.split("_")
        if len(ls) >1:
            for w in ls:
                keep.apped(w)
        else:
            keep.append(word)
    sent = ' '.join(word.lower() for word in keep)
    return sent




#the working path is wheere I take the files with the articles
working_path = 'C:/Users/nikob/Desktop/UnUtrecht/EmpathyDetection/code/SecondRQ/TwitterReddit/data/articlesRedditPopular/'

saving_path = 'C:/Users/nikob/Desktop/UnUtrecht/EmpathyDetection/code/SecondRQ/TwitterReddit/data/commentsTwitter5/'

for entry in reversed(list(os.listdir(working_path))):
    if(entry[0] == "c"):
        continue
    if(entry in list(os.listdir(saving_path))):
        continue
    print(entry)
    file = pd.read_csv(working_path+entry)
    tw_texts = []
    tw_dates = []
    tw_ids = []
    #tw_reply_st = []
    #tw_username = []
    #tw_retw_counts = []
    #tw_favourites = []
    tw_subreddit = []
    tw_articleurl = []
    tw_articleid = []
    for index, row in file.iterrows():
        redditdateposted = datetime.datetime.strptime(row.date[:10], '%Y-%m-%d').date()
        print(datetime.date.today() - redditdateposted)
        #if (datetime.date.today() - redditdateposted).days <2:
            #continue
        #datelimit = redditdateposted + datetime.timedelta(days=5)
        articleurl = row.url
        subreddit = entry[:-10]
        redditid = row.id
        title = str(row.title).lower()
        query = articleurl + " -filter:retweets" #" -"+str(title)+  KEEP Retweets
        print("\n")
        print(title)
        print(articleurl)
        tweets = tw.Cursor(api.search, q=query,lang="en",
                           since=redditdateposted,
                           result_type='mixed',tweet_mode="extended").items(1000)
        try:
            for tweet in tweets:
                text = fix_text(tweet.full_text)
                nm = tweet.user.screen_name
                tw_id = tweet.id
                if str(title) not in text and text.split()[0] != "rt":
                    print('1')
                    print(text)
                    tw_texts.append(text)
                    tw_dates.append(tweet.created_at)
                    tw_ids.append(tw_id)
                    #tw_reply_st.append(tweet.in_reply_to_status_id)
                    #nm = tweet.user.screen_name
                    #tw_retw_counts.append(tweet.retweet_count)
                    #tw_favourites.append(tweet.favorite_count)
                    tw_subreddit.append(subreddit)
                    tw_articleurl.append(articleurl)
                    tw_articleid.append(redditid)
                else:
                    print('2')
                replies = tw.Cursor(api.search, q='to:{}'.format(nm), lang="en",since_id=tw_id,
                                    tweet_mode="extended").items(1000)
                try:
                    for reply in replies:
                        if reply.in_reply_to_status_id == tw_id:
                            print("REPLY \n")
                            tw2_id = reply.id
                            tw2_tx = fix_text(reply.full_text)
                            print(tw2_tx)
                            tw_texts.append(tw2_tx)
                            tw_dates.append(reply.created_at)
                            tw_ids.append(tw2_id)
                            # tw_reply_st.append(tweet.in_reply_to_status_id)
                            #nm = tweet.user.screen_name
                            # tw_retw_counts.append(tweet.retweet_count)
                            #tw_favourites.append(reply.favorite_count)
                            tw_subreddit.append(subreddit)
                            tw_articleurl.append(articleurl)
                            tw_articleid.append(redditid)
                except:
                    print('oups2')
        except:
            print('oups')
        #with title only
        query = str(title) + " -filter:retweets"
        tweets = tw.Cursor(api.search, q=query, lang="en",
                           since=redditdateposted,
                           result_type='mixed', tweet_mode="extended").items(1000)
        try:
            for tweet in tweets:
                tw_id = tweet.id
                if tw_id in tw_ids:
                    continue
                text = fix_text(tweet.full_text)
                nm = tweet.user.screen_name
                if str(title) not in text and text.split()[0] != "rt":
                    print('3')
                    tw_texts.append(text)
                    tw_dates.append(tweet.created_at)
                    tw_ids.append(tw_id)
                    # tw_reply_st.append(tweet.in_reply_to_status_id)
                    # nm = tweet.user.screen_name
                    # tw_retw_counts.append(tweet.retweet_count)
                    # tw_favourites.append(tweet.favorite_count)
                    tw_subreddit.append(subreddit)
                    tw_articleurl.append(articleurl)
                    tw_articleid.append(redditid)
                else:
                    print('4')
                replies = tw.Cursor(api.search, q='to:{}'.format(nm), lang="en", since_id=tw_id,
                                    tweet_mode="extended").items(1000)
                try:
                    for reply in replies:
                        if reply.in_reply_to_status_id == tw_id:
                            print("REPLY \n")
                            tw2_id = reply.id
                            tw2_tx = fix_text(reply.full_text)
                            print(tw2_tx)
                            tw_texts.append(tw2_tx)
                            tw_dates.append(reply.created_at)
                            tw_ids.append(tw2_id)
                            # tw_reply_st.append(tweet.in_reply_to_status_id)
                            # nm = tweet.user.screen_name
                            # tw_retw_counts.append(tweet.retweet_count)
                            # tw_favourites.append(reply.favorite_count)
                            tw_subreddit.append(subreddit)
                            tw_articleurl.append(articleurl)
                            tw_articleid.append(redditid)
                except:
                    print('oups3')
        except:
            print('oups4')
    df = pd.DataFrame(list(zip(tw_ids, tw_texts, tw_dates,tw_subreddit,tw_articleurl, tw_articleid )),
                             columns=['id', 'text', 'date', 'subreddit','url','articleurl'])
    name = entry.split(".")[0]
    pt = saving_path + name + ".csv"
    df.to_csv(pt)

