{ pkgs, lib, config, inputs, ... }:

{
  packages = [ pkgs.git pkgs.kaggle pkgs.cudatoolkit ];

  languages.python.enable = true;
  languages.python.venv.enable = true;
  languages.python.venv.requirements = ./requirements.txt;
}
