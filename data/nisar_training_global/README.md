# Expanded training inputs

53 training examples: Sabancaya 10, Ubinas 9, Misti 10, Mount Asama 3,
Popocatepetl 11, Dabbahu 10. Twelve reporting examples: 2 each from the
three Peru volcanoes and 3 each from Mexico and Ethiopia. Japan is training only.

Train/validation archives for four volcanoes are copies of their original
prepared inputs, recorded in input_sources.json. Ubinas was rebuilt from the
existing native subsets using ubinas_clean_config.json, local-only preparation,
and maximum baseline 24 days; two BETA pairs were excluded. Its original
archives remain unchanged. Only provisional NISAR GUNW is used here.

The runner adds Japan's ascending and descending examples explicitly, sharing
one physical-volcano group. Reporting windows are excluded from fitting and
configuration selection. Temporal overlap across sites makes this retrospective.
