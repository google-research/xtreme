# MultiCheckList

MultiCheckList consists of six tests of multilingual models
for question answering in 50 languages. It is based on the
[CheckList](https://github.com/marcotcr/checklist) framework
[(ACL 2020)](https://www.aclweb.org/anthology/2020.acl-main.442.pdf). Each test
consist of a large number of test cases that are automatically generated based
on test templates.

Test templates have been professionally translated from English into the
following 49 languages (shown with their ISO 691-1 code for brevity):
af, ar,az, bg, bn, de, el, es, et, eu, fa, fi, fr,gu, he, hi,ht, hu, id, it,
ja, jv, ka, kk, ko, t, ml, mr, ms, my, nl, pa,pl, pt,qu,ro, ru, sw, ta, te, th,
tl, tr, uk, ur, vi,wo, yo, zh.

We welcome extensions of the tests to new capabilities and other languages.

## Setup instructions

1.  Create a virtual Python environment using anaconda:
```
conda create -n checklist python=3.6
conda activate checklist
```

2.  Install the dependencies via `pip`:
```
pip install absl-py
pip install checklist
```

3. Download the MultiCheckList tests and model predictions from [here](https://pantheon.corp.google.com/storage/browser/xtreme_translations/MultiCheckList) into this folder. [`gsutil`](https://cloud.google.com/storage/docs/gsutil_install) is
the preferred way to download all files at once:

```
gsutil -m cp -r "gs://xtreme_translations/MultiCheckList/lang_suites" .
gsutil -m cp -r "gs://xtreme_translations/MultiCheckList/lang_predictions" .
```

## Overview

### Types of tests

The current framework covers six tests based on [(Ribeiro et al., 2020)](https://www.aclweb.org/anthology/2020.acl-main.442.pdf).
These tests cover the following types of natural language understanding:

- **Comparisons**. A is COMP than B. Who is more / less COMP? Example: Jerry is smarter than Gary. Who is less smart? A: Gary.
- **Intensifiers**. Intensifiers (very, super, extremely) and reducers (somewhat, kinda, etc). Example: Chris is really confident about the project. Sam is confident about the project. Who is most confident about the project? A: Chris.
- **Properties**. Size, shape, age, color. Example: There is a table in the room. The table is green and oval. What color is the table? A: Green.
- **Profession vs nationality**. Example: John is a journalist and German. What is John's profession? A: journalist.
- **Animal vs vehicle**. Example: Example: Philip has an iguana and a minivan. What animal does Philipp have? A: iguana.
- **Animal vs Vehicle v2**. Example: Sharon bought a car. Florence bought a guinea pig. Who bought an animal? A: Florence.

### Testing of models

We provide a script `test_models.py` that can be used to evaluate versions of
mBERT and XLM-R Large that have been fine-tuned on English SQuAD v1.1 on
MultiCheckList.

The script can be run with the following command:
```
python test_models.py --model_name MODEL_NAME [--model_path MODEL_PATH]
```
where `MODEL_NAME` is either `mbert` or `xlmr` and `MODEL_PATH` is the path of
the directory of the fine-tuned model.
If no model path is provided, the predictions stored in `lang_predictions` will
be evaluated.

To evaluate your own model, simply modify the script. The easiest way is to
load your model as part of a `transformers.QuestionAnsweringPipeline` similar
to mBERT and XLM-R.

Alternatively, it is sufficient to create a function that returns the
predictions and confidence values of your model for each example (see
`predconfs` in the provided script).

The script will print a summary for each test and each language including
example failure cases. The predictions of each test case in each language are
written to a folder (`lang_predictions` by default) for further analysis.
After all tests have been evaluated, the script writes the results for all tests
to a .csv file.

### Generation of tests

The script `generate_tests.py` is used to generate new test cases for each test
and each language.

The script can be run using the following command:
```
python generate_tests.py --save_dir SAVE_DIR --num_test_samples NUM_SAMPLES
```
where `SAVE_DIR` is the diretory where the CheckList suites should be saved
and `NUM_SAMPLES` is the number of test cases generated per test.

The script takes as input the multilingual templates in
`checklist_templatse.tsv`. By default, `SAVE_DIR` is set to `lang_suites` and running the script overwrites
the files in `SAVE_DIR`.

### Extension to new capabilities and languages

In order to extend the tests to new capabilities or new languages, you would need to modify the following files:

- `checklist_templates.tsv`: Here, you would need to add separate columns for new languages or separate rows for new tests.
Note that some languages where disambiguation of certain placeholders is required may need multiple columns.
- `generate_tests.py`: This script should only require modification if new tests are added.
- `generate_test_utils.py`: You would need to add normalisation of articles and punctuation in the new language to the `clean` function.
In addition, if the new language includes multiple variations of a template, then this would need to be handled in the method associated
with the respective test.
- `test_models.py`: This file only needs to be modified when adding new capabilities.
