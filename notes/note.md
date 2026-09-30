say which $ASR$ you are using with the corresponding literature
problem of NLI : if we loose information trough anonymisation, then we won't see it with the model since A -> B still.

3. Le gros problème : Generalization

Là, les modèles divergent fortement.

Exemple :

"I moved to Geneva in 2022."

→

"I moved to Switzerland a few years ago."

C'est typiquement ce que pourrait faire une anonymisation :

specific identity-related information
                ↓
      more general information
Base
$$ P(E)=0.182 $$ $$ P(N)=0.795 $$ $$ P(C)=0.023 $$

Donc il dit essentiellement :

neutral

Ce qui est raisonnable.

Large
$$ P(E)=0.022 $$ $$ P(N)=0.105 $$ $$ P(C)=0.874 $$

Et là il dit :

contradiction

Ce qui est beaucoup moins raisonnable : déménager à Genève implique déménager en Suisse, donc les phrases sont compatibles.




The goal is to preserve the semantic content of the interview while facing two modelling challenges: (1) it is difficult to quantify how much semantic information is retained after anonymization, and (2) it is context-dependent which words should be replaced or preserved—for example, “I live in Lausanne” may need to be changed to Geneva when Lausanne uniquely identifies a speaker, but should be preserved when it does not provide distinguishing information, as changing it would only unnecessarily alter the content.


VoiceChallenge2026 uses ECAPA-TDNN like us


