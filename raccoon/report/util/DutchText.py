from raccoon.report.util.BaseText import BaseText

class DutchText(BaseText):
    def __init__(self):
        super().__init__(
            intro_text=(
                "Dit rapport vergelijkt {retriever_count} {retriever_word} voor retrieval-augmented generation. "
                "Het geeft een samenvatting van de corpus- en queryset, vergelijkt de rankingkwaliteit en runtime, "
                "en toont vervolgens de configuratie en voorbeeldresultaten van elke retriever. Gebruik de "
                "vergelijkingssectie om te bepalen welke retriever het meest waarschijnlijk nuttig bewijs in "
                "het contextvenster van de generator plaatst."
            ),

            metric_guide_text=(
                "<b>Uitleg metrics:</b> Deze scores evalueren de retrievalfase voordat het taalmodel een antwoord genereert. "
                "Binnen een RAG-pipeline betekent betere retrieval dat de generator relevanter en beter gerankt bewijs ontvangt "
                "en minder hoeft te vertrouwen op niet-onderbouwde modelkennis."
            ),

            metric_text=(
                "<b>Recall@k</b> meet dekking: hoeveel relevante documenten zijn gevonden binnen de top k. "
                "<b>Precision@k</b> meet focus: hoeveel van de top k daadwerkelijk relevant is in plaats van afleidende context. "
                "<b>MAP@k</b> beloont relevante documenten die consistent vroeg verschijnen over meerdere queries heen. "
                "<b>NDCG@k</b> beloont zowel de rangorde als graduele relevantie en is daarom vaak de beste enkele indicator "
                "voor de vraag of het sterkste bewijs als eerste de prompt bereikt."
            ),
            practical_interpretation_text=(
                "<b>Praktische interpretatie:</b> Gebruik deze resultaten niet alleen om de hoogste score te kiezen, "
                "maar om te bepalen welke retriever het beste past bij het doel van de RAG-toepassing. "
                "Een hoge Recall@k is vooral belangrijk wanneer het systeem zo min mogelijk relevante informatie mag missen, "
                "bijvoorbeeld bij juridische of ondersteunende zoekvragen. Een hoge Precision@k is belangrijk wanneer het "
                "contextvenster beperkt is en irrelevante passages de generator kunnen afleiden. "
                "Wanneer reranking duidelijk betere NDCG- of MAP-scores geeft, betekent dit dat de juiste documenten niet alleen "
                "gevonden worden, maar ook hoger in de ranglijst komen te staan. Dat is praktisch waardevol omdat een taalmodel "
                "meestal sterker leunt op de eerste passages in de context. "
                "Runtime moet daarnaast worden meegewogen: een iets lagere score kan acceptabel zijn wanneer de retriever veel "
                "sneller is of eenvoudiger te onderhouden blijft."
            ),

            throughput_text=(
                "<b>Uitleg runtime en throughput:</b> Deze resultaten evalueren de praktische prestaties van de retrievalfase "
                "naast retrievalkwaliteit. Query time meet hoeveel tijd nodig is om queries te verwerken, terwijl index time "
                "aangeeft hoeveel tijd nodig is om de volledige index op te bouwen. Queries/sec en documents/sec meten de "
                "throughput van het systeem en geven inzicht in schaalbaarheid binnen grotere RAG-omgevingen. Storage MB toont "
                "hoeveel opslagruimte de retrievalindex gebruikt, wat belangrijk kan zijn binnen productieomgevingen waar "
                "infrastructuurkosten en geheugenverbruik een rol spelen. De grafiek combineert retrievalkwaliteit met throughput "
                "door Recall@10 af te zetten tegen queries per seconde. Hogere posities betekenen betere retrievaldekking, "
                "terwijl posities verder naar rechts wijzen op hogere snelheid en lagere latency."
            ),

            throughput_interpretation_text=(
                "<b>Praktische interpretatie:</b> Gebruik deze resultaten niet alleen om de snelste of meest accurate retriever "
                "te kiezen, maar om te bepalen welke retrievalarchitectuur het beste aansluit bij de praktische eisen van de "
                "toepassing. Een retriever rechtsboven in de grafiek combineert hoge retrievalkwaliteit met hoge throughput en "
                "vormt daardoor meestal de beste algemene balans voor real-time RAG-systemen. Een retriever kan echter bewust "
                "meer querytijd gebruiken om betere ranking of hogere retrievaldekking te behalen. In productieomgevingen moet "
                "daarom rekening worden gehouden met latency, schaalbaarheid, hardwarekosten en onderhoudscomplexiteit. "
                "Een iets lagere retrievalscore kan acceptabel zijn wanneer een systeem aanzienlijk sneller, goedkoper of "
                "eenvoudiger schaalbaar blijft."
            ),
        )