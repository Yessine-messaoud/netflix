# Task 3 - Recommendation results

## Because you watched *Midnight Mass*

| title                      | type    | listed_in                          | director   | country       |   similarity |
|:---------------------------|:--------|:-----------------------------------|:-----------|:--------------|-------------:|
| Brand New Cherry Flavor    | TV Show | TV Dramas, TV Horror, TV Mysteries | Unknown    | United States |        0.79  |
| Ratched                    | TV Show | TV Dramas, TV Horror, TV Mysteries | Unknown    | United States |        0.79  |
| The Haunting of Bly Manor  | TV Show | TV Dramas, TV Horror, TV Mysteries | Unknown    | United States |        0.79  |
| The Haunting of Hill House | TV Show | TV Dramas, TV Horror, TV Mysteries | Unknown    | United States |        0.79  |
| The Originals              | TV Show | TV Dramas, TV Horror, TV Mysteries | Unknown    | United States |        0.781 |

## Because you watched *Stranger Things*

| title                          | type    | listed_in                                       | director   | country       |   similarity |
|:-------------------------------|:--------|:------------------------------------------------|:-----------|:--------------|-------------:|
| Chilling Adventures of Sabrina | TV Show | TV Horror, TV Mysteries, TV Sci-Fi & Fantasy    | Unknown    | United States |        1     |
| Helix                          | TV Show | TV Horror, TV Mysteries, TV Sci-Fi & Fantasy    | Unknown    | United States |        0.991 |
| Nightflyers                    | TV Show | TV Horror, TV Mysteries, TV Sci-Fi & Fantasy    | Unknown    | United States |        0.991 |
| Anjaan: Special Crimes Unit    | TV Show | International TV Shows, TV Horror, TV Mysteries | Unknown    | India         |        0.757 |
| The Originals                  | TV Show | TV Dramas, TV Horror, TV Mysteries              | Unknown    | United States |        0.757 |

## Because you watched *Narcos*

| title                  | type    | listed_in                                        | director   | country       |   similarity |
|:-----------------------|:--------|:-------------------------------------------------|:-----------|:--------------|-------------:|
| Marvel's Luke Cage     | TV Show | Crime TV Shows, TV Action & Adventure, TV Dramas | Unknown    | United States |            1 |
| Marvel's Iron Fist     | TV Show | Crime TV Shows, TV Action & Adventure, TV Dramas | Unknown    | United States |            1 |
| Marvel's The Defenders | TV Show | Crime TV Shows, TV Action & Adventure, TV Dramas | Unknown    | United States |            1 |
| Marvel's Jessica Jones | TV Show | Crime TV Shows, TV Action & Adventure, TV Dramas | Unknown    | United States |            1 |
| Altered Carbon         | TV Show | Crime TV Shows, TV Action & Adventure, TV Dramas | Unknown    | United States |            1 |

## Because you watched *The Irishman*

| title             | type   | listed_in                        | director        | country       |   similarity |
|:------------------|:-------|:---------------------------------|:----------------|:--------------|-------------:|
| Gangs of New York | Movie  | Dramas                           | Martin Scorsese | United States |        1     |
| The Departed      | Movie  | Dramas, Thrillers                | Martin Scorsese | United States |        0.872 |
| Hugo              | Movie  | Children & Family Movies, Dramas | Martin Scorsese | United States |        0.854 |
| Shutter Island    | Movie  | Thrillers                        | Martin Scorsese | United States |        0.806 |
| GoodFellas        | Movie  | Classic Movies, Dramas           | Martin Scorsese | United States |        0.78  |

## Because you watched *Ganglands*

| title         | type    | listed_in                                                     | director   | country   |   similarity |
|:--------------|:--------|:--------------------------------------------------------------|:-----------|:----------|-------------:|
| Lupin         | TV Show | Crime TV Shows, International TV Shows, TV Action & Adventure | Unknown    | France    |        0.727 |
| Crime Time    | TV Show | Crime TV Shows, International TV Shows, TV Action & Adventure | Unknown    | France    |        0.727 |
| Fatal Destiny | TV Show | Crime TV Shows, International TV Shows, TV Action & Adventure | Unknown    | Pakistan  |        0.669 |
| Agent Raghav  | TV Show | Crime TV Shows, International TV Shows, TV Action & Adventure | Unknown    | India     |        0.666 |
| Smoking       | TV Show | Crime TV Shows, International TV Shows, TV Action & Adventure | Unknown    | Japan     |        0.663 |

## Evaluation (500 random queries, top-10)

|                    |   Content model |   Random baseline |
|:-------------------|----------------:|------------------:|
| Genre Jaccard@10   |           0.831 |             0.09  |
| Genre hit-rate@10  |           0.985 |             0.231 |
| Same type@10       |           0.994 |             0.586 |
| Catalogue coverage |           0.23  |           nan     |
