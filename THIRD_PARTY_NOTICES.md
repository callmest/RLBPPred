# Third-party sources

The original package and strict MLP weights are distributed by callmest under
the MIT license. They do not redistribute SaProt or Foldseek weights/binaries
inside the wheel or Git tree. The explicit setup command retrieves fixed files
from their upstream publishers; their original terms apply.

- [SaProt source](https://github.com/westlake-repl/SaProt) and
  [fixed encoder repository](https://huggingface.co/westlake-repl/SaProt_650M_AF2/tree/1b8cddd7bad34f6f5627422b4908309840c78fef).
  SaProt's source is MIT licensed. Its AA+3Di alphabet/loading convention is
  implemented here for representation compatibility. The unused contact
  prediction head is omitted; all embedding-producing parameters load strictly.
- [fair-esm](https://github.com/facebookresearch/esm), MIT, version 2.0.0;
  the separately named ESM-C `esm` package is not a substitute.
- [Foldseek](https://github.com/steineggerlab/foldseek), GPL-3.0;
  [license](https://github.com/steineggerlab/foldseek/blob/10-941cd33/LICENSE.md).
  It runs as an external executable, pinned to release 10-941cd33 by setup.
- [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/): structures
  and metadata are retrieved from the official service and retain its terms
  and provenance. Cite the upstream work when using these representations or
  structures in a publication.

SaProt source notice:

MIT License — Copyright (c) 2023 westlake-repl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
