{ pkgs, lib, config, inputs, ... }:

{
  # Local box has an NVIDIA RTX 2050. The system driver's libcuda.so lives at
  # /usr/lib/x86_64-linux-gnu; export it so the pip CUDA torch build can load
  # it (mirrors ../roco_ai/devenv.nix). Without this, torch.cuda.is_available()
  # is False even though the driver is present.
  env.LD_LIBRARY_PATH = "/usr/lib/x86_64-linux-gnu";

  packages = [ pkgs.git pkgs.kaggle pkgs.cudatoolkit ];

  languages.python.enable = true;
  languages.python.venv.enable = true;
  languages.python.venv.requirements = ./requirements.txt;
}
