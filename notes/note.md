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
