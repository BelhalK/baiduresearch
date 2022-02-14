\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{alt2022}[2022/02/07 v1.1 ALT 2022 Proceedings]
\newcommand{\todo}{\color{red}}

\@ifundefined{ifanonsubmission}{
  \newif\ifanonsubmission
  \anonsubmissionfalse
}{}

\@ifundefined{iffinalsubmission}{
  \newif\iffinalsubmission
  \finalsubmissionfalse
}{}

\DeclareOption{anon}{\anonsubmissiontrue}
\DeclareOption{final}{\finalsubmissiontrue}
\ProcessOptions\relax

\LoadClass[pmlr]{jmlr}

\jmlrvolume{vol 167}
\jmlryear{2022}
\jmlrproceedings{PMLR}{Proceedings of Machine Learning Research}

\ifanonsubmission
 \newcommand{\altauthor}[1]{}
 \author{author names withheld}
 \editor{Under Review for ALT 2022}
 \jmlrworkshop{33rd International Conference on Algorithmic Learning Theory}
 \renewcommand{\acks}[1]{}
\else
 \newcommand{\altauthor}[1]{\author{#1}}
 \iffinalsubmission
  \editors{Sanjoy Dasgupta and Nika Haghtalab}
  \jmlrworkshop{33rd International Conference on Algorithmic Learning Theory}
 \fi
\fi



