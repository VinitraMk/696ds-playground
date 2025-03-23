import re

def __extract_numbered_bullets(text, add_newline=False):
        bullet_pattern = r"^\s*\d+[\.\)-]\s+"
        lines = text.split("\n")
        numbered_bullets = [re.sub(bullet_pattern, "", line).strip() + ("\n" if add_newline else "")
                            for line in lines if re.match(bullet_pattern, line)]
        return numbered_bullets


text = """1. In 2023, 2022, and 2021, the company reported earnings increases for their investments totaling \$16M, \$14M, and \$6M consecutively.
   
2. By end-of-year reports up until Decemeber 30th, 2023; there has been no change noted regarding receivable amounts owed specifically towards joint venture projects involving THASIC since they remain null across both fiscal periods mentioned here i.e., ending dates being either december last day before newyear OR afternewyearsday depending if leap yr applies not sure though but just saying "as per records" so yeah same thing really lol jkjk actually meant decembe rlast days pre post ny etc ok moving forward...

   *Note*: This sentence seems overly complex when summarizing facts about financial statements typically would be more straightforward like below instead...
  
    **Corrected**: 
      
      There havebeenno outstandingreceivalblebalancesfromtheCompany’sinvestmentinthethATIcJVastheendsofbothfinancialperiodsmentioned,i\.e\(.\)\,\Decemberr \(\) \(\[\\\[\\\]$$)$er $$ndingdatesbeingeitherdecemberlasdtbeforeorafterthenewyeardaydependingifleapyrapplesnotsurethoughbutjustsayingsasperrecords”actuallymeantdecmbrldaysprepostnyetcokmovingforward...**

     Correctly simplified version should look something along these lines without unnecessary complications & redundancies introduced earlier accidentally during initial attempt above : 

       Nooutstanding balanceswereowedbythatjcjvfortherecordendingonbothsaiddatesthatwerereporteduponherei,e,endoffiscalcalendarwithrespecttodecemberspecificallymentionedinthiscontextwhichwouldmeantheyarezeroatthesetimespanningacrossrelevanttimeframesgivenwithinprovidedinformationaboveaboutcompanyfinancesrelatedtocertaininvestmentsmadeintootherentitieslikepartnerships/jointventuresetccoveredsomeportionoftextsharedinitialywhenrequestwasfirstmadeforthistasktoprocessandsimplifydownintomoresuccinctformattingstyleconsistentwitheachindividualpointtryingtomakeclearwithoutanyextraneousdetailsincludedunnecessarilymakingitmorecomplicatedthanneedbetobeproperlysimplifiedversionforthegiveninputdataaboutrawfactscontainedinitoriginallinesuppliedduringoriginal
"""
print(text)

print(__extract_numbered_bullets(text))