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


#text = """1. In 2023, 2022, and 2021, the company reported earnings increases for their investments totaling \$16M, \$14M, and \$6M consecutively.
   
#2. By end-of-year reports up until Decemeber 30th, 2023; there has been no change noted regarding receivable amounts owed specifically towards joint venture projects involving THASIC since they remain null across both fiscal periods mentioned here i.e., ending dates being either december last day before newyear OR afternewyearsday depending if leap yr applies not sure though but just saying "as per records" so yeah same thing really lol jkjk actually meant decembe rlast days pre post ny etc ok moving forward...

   #*Note*: This sentence seems overly complex when summarizing facts about financial statements typically would be more straightforward like below instead...
  
    #**Corrected**: 
      
      #There havebeenno outstandingreceivalblebalancesfromtheCompany’sinvestmentinthethATIcJVastheendsofbothfinancialperiodsmentioned,i\.e\(.\)\,\Decemberr \(\) \(\[\\\[\\\]$$)$er $$ndingdatesbeingeitherdecemberlasdtbeforeorafterthenewyeardaydependingifleapyrapplesnotsurethoughbutjustsayingsasperrecords”actuallymeantdecmbrldaysprepostnyetcokmovingforward...**

     #Correctly simplified version should look something along these lines without unnecessary complications & redundancies introduced earlier accidentally during initial attempt above : 

       #Nooutstanding balanceswereowedbythatjcjvfortherecordendingonbothsaiddatesthatwerereporteduponherei,e,endoffiscalcalendarwithrespecttodecemberspecificallymentionedinthiscontextwhichwouldmeantheyarezeroatthesetimespanningacrossrelevanttimeframesgivenwithinprovidedinformationaboveaboutcompanyfinancesrelatedtocertaininvestmentsmadeintootherentitieslikepartnerships/jointventuresetccoveredsomeportionoftextsharedinitialywhenrequestwasfirstmadeforthistasktoprocessandsimplifydownintomoresuccinctformattingstyleconsistentwitheachindividualpointtryingtomakeclearwithoutanyextraneousdetailsincludedunnecessarilymakingitmorecomplicatedthanneedbetobeproperlysimplifiedversionforthegiveninputdataaboutrawfactscontainedinitoriginallinesuppliedduringoriginal
#"""

text = """
system

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are a helpful assistant, that given a list of factoids, generates meaningful and complex questions from it.user

### Task:
        Given the list of factoids below and metadata, generate a complex question that requires reasoning over multiple factoids

        ### Generation Rules
        - **Do not use chinese characters** in your response. Return responses in English only.
        - Keep the generated query under 100 words.
        - **Do not put gibberish, unnecessary and ellaborate adjectives** in your response for either question or the answer.
        - **Do not put intermediate, thinking or reasonings steps in your response**
        - Don't think for more than 2000 tokens
        - Use the example structure to return the final response.
        - **Do not copy example from the prompt** in your response.

        ### Input format:
        Metadata: <meta data of the main company upon which the factoids are based.>
        Factoids: [\<list of factoids\>]

        ### Output format:
        Query: {
        "query": <question generated from fact(s) in the given text document>
        }

        ### Example Input
        Metadata: Company name: Apple | SEC Filing: 10-K | Related Topic: Risk Factors and Challenges
        Factoids: {["Apple committed to carbon neutrality across its supply chain by 2030.",
            "Apple sources renewable energy for its global operations.",
            "Apple integrates recycled materials into product design.",
            "The company works with suppliers to reduce emissions.",
            "Upfront costs have increased due to sustainability investments."
        ]}

        ### Example Output:
        Query:
        {
            "query": "How does Apple’s commitment to achieving carbon neutrality across its supply chain and products by 2030, as discussed in its 10-K, affect its cost structure, supplier relationships, and long-term profitability, and what are the potential risks and rewards associated with this aggressive ESG strategy?",
        }

        ### Input for your task:
        
Metadata: Company: Nvidia Corp. | SEC Filing: 10-K | Related topic: Risk Factors and Challenges
Factoids: [NVIDIA aims to include underrepresented individuals such as women, Black/African Americans, and Hispanic/Latinos in recruitment and development initiatives.,
NVIDIA partners with educational and professional organizations focused on historically underrepresented communities.,
Dedicated recruiting teams embed themselves into different departments to guide underrepresented candidates during interviews and identify job openings.,
Training sessions are conducted for managers and colleagues to foster inclusive workplaces and promote diversity in recruitment.,
At the close of fiscal year 2024, NVIDIA had 79% males, 20% females, and 1% non-declared gender representation globally.,
Stakeholders might initiate legal actions over inadequate responsiveness towards climate change.,
Failure to meet sustainability goals could lead to reputational damage and unforeseen expenses.,
Climate change-induced supply chain issues may cause contract disagreements leading to higher litigation and costs.,
High energy consumption associated with GPUs poses a risk under growing environmental scrutiny.,
Inability to fully benefit from business investments or integrations post-acquisition impacts growth strategies.,
Acquisitions come with integration difficulties potentially harming financial performance.,
Regulatory hurdles delay acquisitions risking loss of talent and reduced expected returns.,
Failure of invested companies could lead to recognition of impairments or loss of investment.,
Investment portfolio faces industry sector concentration risks leading to higher impairment losses.,
Acquisitions pose risks such as resource diversion, realization uncertainty, and legal proceedings.,
Integration challenges during acquisitions affect technology, systems, product policies, employee retention.,
Assuming liabilities post-acquisition leads to amortization expenses and asset impairment charges.,
Stock price impact occurs if regulatory approval cannot be obtained for acquisitions.,
Issuance of debt for acquisitions increases overall debt levels and associated costs.,
Negative economic conditions in targeted regions or industries can negatively influence acquisitions' outcomes.,
System integrations after acquisitions can cause operational disruptions and cost overruns.,
Sales to a few major partners significantly contribute to revenue exposure.,
Customer A contributed 13% to total revenue in FY2024 under the Compute & Networking segment.,
Increased supply and capacity purchases with existing suppliers add complexity and execution risk.,
Shortened component lead times affected inventory and manufacturing capacity purchase commitments at the end of fiscal year 2024.]assistant

Query: 
{
    "query": "How do Nvidia's diversity and inclusion initiatives, and its exposure to climate change and acquisition risks, impact its long-term financial performance and reputation, considering the potential consequences of regulatory hurdles, integration challenges, and concentration risks in its investment portfolio?"
}
"""
#print(text)
ti = text.index('Input for your task')
print(text[ti:])
#print(__extract_numbered_bullets(text))
print(extract_json_text_by_key(text[ti:], "query"))