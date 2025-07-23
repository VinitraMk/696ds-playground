import re
import json

def __extract_numbered_bullets(text, add_newline=False):
        bullet_pattern = r"^\s*\d+[\.\)-]\s+"
        lines = text.split("\n")
        numbered_bullets = [re.sub(bullet_pattern, "", line).strip() + ("\n" if add_newline else "")
                            for line in lines if re.match(bullet_pattern, line)]
        return numbered_bullets

def extract_json_text_by_key(raw_text, target_key):
    """
    Searches raw text for a JSON object that contains a specific key
    and returns it as a Python dictionary. Returns None if not found.
    """
    # Match any JSON object containing the key: { "target_key": "some_value" }
    pattern = rf'\{{[^{{}}]*"{re.escape(target_key)}"\s*:\s*"[^"{{}}]*"[^{{}}]*\}}'

    match = re.search(pattern, raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

def extract_json_object_by_key(text: str, key: str):
    """
    Extract a JSON/dictionary object from a string given a top-level key like 'evaluation'.
    Returns the parsed dictionary or None if not found or malformed.
    """
    # Create a regex pattern that captures the content of the object after the key
    pattern = rf'"{re.escape(key)}"\s*:\s*{{.*?}}'  # non-greedy match
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    try:
        # Add braces to make it a valid JSON string if needed
        json_fragment = '{' + match.group(0) + '}'
        parsed = json.loads(json_fragment)
        return parsed.get(key)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None

#text = """1. In 2023, 2022, and 2021, the company reported earnings increases for their investments totaling \$16M, \$14M, and \$6M consecutively.
   
#2. By end-of-year reports up until Decemeber 30th, 2023; there has been no change noted regarding receivable amounts owed specifically towards joint venture projects involving THASIC since they remain null across both fiscal periods mentioned here i.e., ending dates being either december last day before newyear OR afternewyearsday depending if leap yr applies not sure though but just saying "as per records" so yeah same thing really lol jkjk actually meant decembe rlast days pre post ny etc ok moving forward...

   #*Note*: This sentence seems overly complex when summarizing facts about financial statements typically would be more straightforward like below instead...
  
    #**Corrected**: 
      
      #There havebeenno outstandingreceivalblebalancesfromtheCompany’sinvestmentinthethATIcJVastheendsofbothfinancialperiodsmentioned,i\.e\(.\)\,\Decemberr \(\) \(\[\\\[\\\]$$)$er $$ndingdatesbeingeitherdecemberlasdtbeforeorafterthenewyeardaydependingifleapyrapplesnotsurethoughbutjustsayingsasperrecords”actuallymeantdecmbrldaysprepostnyetcokmovingforward...**

     #Correctly simplified version should look something along these lines without unnecessary complications & redundancies introduced earlier accidentally during initial attempt above : 

       #Nooutstanding balanceswereowedbythatjcjvfortherecordendingonbothsaiddatesthatwerereporteduponherei,e,endoffiscalcalendarwithrespecttodecemberspecificallymentionedinthiscontextwhichwouldmeantheyarezeroatthesetimespanningacrossrelevanttimeframesgivenwithinprovidedinformationaboveaboutcompanyfinancesrelatedtocertaininvestmentsmadeintootherentitieslikepartnerships/jointventuresetccoveredsomeportionoftextsharedinitialywhenrequestwasfirstmadeforthistasktoprocessandsimplifydownintomoresuccinctformattingstyleconsistentwitheachindividualpointtryingtomakeclearwithoutanyextraneousdetailsincludedunnecessarilymakingitmorecomplicatedthanneedbetobeproperlysimplifiedversionforthegiveninputdataaboutrawfactscontainedinitoriginallinesuppliedduringoriginal
#"""

text = """
Answer: {
    "answer": "The USG's export controls could significantly influence Nvidia's manufacturing strategies and partnerships. The controls may limit alternative manufacturing locations and encourage 'design-out' of US semiconductors. The USG has imposed licensing requirements on Nvidia's products, including A100, A800, H100, H800, and L40S, which has directly impacted Nvidia's business operations. The USG may change export control rules at any time, subjecting a wider range of products to export restrictions and licensing requirements, which could harm Nvidia's competitive position and financial results. Compliance with the USG's export controls could increase Nvidia's costs and impact its competitive position. Excessive or shifting export controls may encourage customers to 'design-out' certain US semiconductors from their products, and overseas governments may request that customers purchase from competitors rather than Nvidia. The USG's export controls may also limit Nvidia's ability to sell products to certain countries, including China, which has already impacted Nvidia's Data Center revenue. Changes in the USG's export controls may require Nvidia to change its business practices, which could negatively impact its financial results. Nvidia may seek a license from the USG for products covered by licensing requirements, but there is no assurance that the license will be granted or that the USG will act on the license application in a timely manner, which could negatively impact the company's business and financial results."
}

"""

text = """
Evaluation: {
            "evaluation": {
                "entity_relevance": 1,
                "source_faithfulness": 1,
                "key_info_coverage": 1,
                "numeric_recall": 0,
                "non_redundancy": 1,
                "total_score": 4,
                "justification": "The groundings are highly relevant to the entity 'Technologies' and accurately reflect the content of the chunk and metadata. They cover key points such as the importance of adapting to changes in technology, the need for research and development, and the company's efforts to provide innovative technologies. However, they lack specific numeric details, which is the only criterion not fully met."
            }
        }

""" 
print(text)
#ti = text.index('Answer: ') + 8
#print(text[ti:])
#print(__extract_numbered_bullets(text))
print(extract_json_object_by_key(text, "evaluation"))