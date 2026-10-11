# 审计来源与文件索引

本目录保存小型报告、API矩阵、精简JSON、CPU/CUDA XML和小型历史GDB日志。`source_evidence_index.json` 记录原始文件SHA与仓库副本SHA；Markdown仅加入仓库相对路径/打包说明，原始报告保持不变。生成的精简机器记录另外列出SHA。索引自身的复制SHA见外部打包核验收据。

绝对路径是原始审计机器的取证位置；克隆仓库不需要访问它们来读取本目录报告。大源码清单、完整GDB/scene数据保持本机原位置，不把缺失外部大文件当作克隆后已具备。失败和早期测试收据保留真实时间/环境，不能用新成功覆盖原失败。

## 随包报告与收据

| 仓库内文件 | 字节 | 原始SHA256 |
|---|---:|---|
| [ALL_CODE_AND_VENDOR_AUDIT_ZH.md](ALL_CODE_AND_VENDOR_AUDIT_ZH.md) | 7466 | `a1d55756cf09f85a6a1834b90ae233d9579727f7b64667acb22852480b980d47` |
| [HEADLESS_MODAL_SETTING_ZH.md](HEADLESS_MODAL_SETTING_ZH.md) | 1686 | `53bfc5d0de3686c4f7af66b12b823f86594435fbfd2760c204c38cbbe5b9496a` |
| [NATIVE_EXIT_DIAGNOSTIC_ZH.md](NATIVE_EXIT_DIAGNOSTIC_ZH.md) | 7898 | `8d89f9fc69fdbd0906ce001208c382b18729d76519b48febd3ed9585dba93ea0` |
| [NAVDP_API_MIGRATION_ZH.md](NAVDP_API_MIGRATION_ZH.md) | 11718 | `af88a6245fbfac7a1bd78138152e0259d446060b4828433fab66a7a541c24f2e` |
| [NEWTON_MIGRATION_ZH.md](NEWTON_MIGRATION_ZH.md) | 3829 | `a157f3beed55cf68e0ad379e02a42111692ee41788d4db4ebd77bd4b1ba57840` |
| [SDK_MIGRATION_AUDIT_ZH.md](SDK_MIGRATION_AUDIT_ZH.md) | 7892 | `776ba4aa9b9d06ff56b3dee831dfb95ffc37a52b746b5d4759af2c905dc67c33` |
| [agent_changed_source_files.json](agent_changed_source_files.json) | 9369 | `254a23943ce8f89c8dc4d6a6bba1b131c84374be352b90c27ced008f2c99b211` |
| [all_python_duplicate_candidates.json](all_python_duplicate_candidates.json) | 2189 | `8f19717fe16a12884242a0b1882468313aa03bb2ca24bdfe4592c15c24d7b059` |
| [all_python_summary.json](all_python_summary.json) | 890 | `7efd21ec69bbea70affdb996c08cb2f7011a5b9f4aae8c6075d2f517398f00e7` |
| [final_audit_receipt.json](final_audit_receipt.json) | 2168 | `fd6ed8f6707a3dc2df4edb634d117674af56ae299e1656180a8f6f4511214778` |
| [final_newton_vendor_cpu.xml](final_newton_vendor_cpu.xml) | 11424 | `db71050e62ef261c6b08ec0b71466188085f5c73e1fbce355a8e4c79d04c4d17` |
| [headless_iommu_setting_sources.json](headless_iommu_setting_sources.json) | 913 | `443d9118d13125d4fdfc6c9ffb58ba8e2c79922eaa81721d9b33a1c87752bffc` |
| [native_crash_objects.json](native_crash_objects.json) | 8772 | `68e12dbae694bda40ec6d4657cad5b6f0fc7de453ff102a56a2847498d480cc9` |
| [native_gdb.log](native_gdb.log) | 4453 | `4c783715370759e6584c97b039d61e7fedf58a25d2beaf0733f2fb58c9bf7872` |
| [native_gdb_preclose.log](native_gdb_preclose.log) | 2281 | `9087c82fd47e1f9ba54461391a7d2845fef3f2612527b14370f5636ba9676b12` |
| [native_gdb_types.log](native_gdb_types.log) | 8752 | `d0770aa25cac02b8b129f868fc00ee66bc0a3f715ba04cb375756a5f8a61ee7e` |
| [native_n1_exit_acceptance.json](native_n1_exit_acceptance.json) | 2071 | `5c88ebfb5e43e23f4744e7ff7413cb66452c224bc6f08dbdd4208101ee91110c` |
| [navdp_changes.json](navdp_changes.json) | 5011 | `2bad3145cbd063cb86cb0dff72e737ab68e2cd5646bfcfbddf3038e14e0fa55b` |
| [navdp_entry_preflight.json](navdp_entry_preflight.json) | 343 | `7821525f95f80627085671664169f2a491e34a246d8e757e86a58e6dc41d4a0b` |
| [navdp_live_sdk_cpu.json](navdp_live_sdk_cpu.json) | 727 | `13655334b83e761889aeb2cd5953bbfc7413432fc1a8d7c09565186949c39d80` |
| [navdp_matrix_summary.json](navdp_matrix_summary.json) | 109 | `c58f1236eff8ea588219e44e8629fd0be611a643271a18a1b851455b51f2679e` |
| [navdp_namespace_matrix.json](navdp_namespace_matrix.json) | 33894 | `71cbc66e803afde92da047fd2008abd6ccf5aaccb26554db45bd7ccc914f76f0` |
| [navdp_quaternion_changes.json](navdp_quaternion_changes.json) | 4629 | `b3cd7a84f67d26e1b0fb28ef728a2679978ebdf1384a8e40590db500e34b2c36` |
| [navdp_resource_snapshot.json](navdp_resource_snapshot.json) | 466 | `b153f1760318f57ce13f4414082ec3f55afcd21bd3910f3d53cacbc04bef0a24` |
| [newton_all_cpu.xml](newton_all_cpu.xml) | 6371 | `a9d840b0a0a807c4731f3c5323222f447a23ac47e8dda31e5a3c7104afd7b82c` |
| [newton_cpu.xml](newton_cpu.xml) | 2203 | `c21c94dd4aab3703526fb46d809c311a873b5b3a06ec7e6424515a0763f46477` |
| [newton_legacy_cpu.xml](newton_legacy_cpu.xml) | 2203 | `4b2fbb69e7397ab01ffee53619280c0f1eb84d46d58e69a022d8f2d1e94ea692` |
| [newton_wrapper_cpu.json](newton_wrapper_cpu.json) | 998 | `905562ae79b03bb5f5b9603ab9f21b26b65f905fd0132e6e44a13f67b124a940` |
| [project_sdk_constructor_calls.json](project_sdk_constructor_calls.json) | 23230 | `b18c5eb9b02136532e6ebf4a59844ab596ab8c143baaa2a29cddd027fc37f65c` |
| [project_sdk_data_accesses.json](project_sdk_data_accesses.json) | 19556 | `86a5bcdbfaf1d552d281b48ad1cbe6a9081ba03681b58002375d0a45fa30507d` |
| [project_sdk_imports.json](project_sdk_imports.json) | 22833 | `7b94a99a3da4716e2ad5094e4447be3e76bea26a3e0eb4eb124118a3cea4703b` |
| [scene_native_crash_objects.json](scene_native_crash_objects.json) | 66787 | `30bbed1ca15588cac450d871e16a147414a4ec95e742396445fc8ff3236272b5` |
| [static_summary.json](static_summary.json) | 511 | `258a99da2a14e0353546ea4468cbf6a79a8db9ecb741cacd02d6eb33261792b5` |
| [vendor_changes.json](vendor_changes.json) | 3003 | `cfdefd0d47bff9b577a0e080259ed35fc95639d01112b017237e12a37e15edfc` |
| [vendor_cpu.xml](vendor_cpu.xml) | 2704 | `1c2a420bb5df8ac486c1bac63070edb598350ed0df7c6a9660b8908ea55f57a7` |
| [vendor_navdp_cpu.xml](vendor_navdp_cpu.xml) | 5300 | `51c13a813c56622a87bb34d4bca298394a52c613ff2b5867e00289a23f47274c` |
| [vendor_navdp_quaternion_cpu.xml](vendor_navdp_quaternion_cpu.xml) | 6507 | `da9a3e3125afba4a28eb054c614e4db38ddb97851fa62e681cb3103d3986bcf0` |
| [gpu_five_actual.xml](gpu_five_actual.xml) | 968 | `0573ae4c24c82505ef1e15f4c1c48ac7bee18dedad9781f8cc6f3cb5a1afbb53` |

## 原始外部大证据

以下只保留来源位置、大小和SHA；本目录没有复制这些文件。完整机器索引见 [source_evidence_index.json](source_evidence_index.json)。

| 原始路径 | 字节 | SHA256 |
|---|---:|---|
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/SecondFloor_gdb_1step/planning.json` | 12489 | `e5fa306caf2914e6fdb7110032f1e83bbd1282f3919bb7a4b7339e0f5a584f00` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/SecondFloor_gdb_1step/result.json` | 340020 | `270f66a6d27d83a207d05d279ca3a66fd407c7fe330678cacd9c321f4424edfd` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/SecondFloor_gdb_1step_instances/planning.json` | 12502 | `502bbc91690c57fb3c9f715c810ce3cc86ccabb6a32e7cfb17699feb77a39ca4` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/SecondFloor_gdb_1step_instances/result.json` | 340047 | `7bf0c571bb9176921c412d4b919ddb571a3af73a5277337d8bf56745ce0864f9` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/all_python_definitions.json` | 2517618 | `6d99b2dac29ec788b02b420ffd61c0d6a6516614530e7d22260122af2faf5a7e` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/all_python_imports.json` | 1843849 | `0df636630f586722b0893747e74ef942567dff24d02bb52fbf8dde6e4c0eea6a` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/all_python_inventory.json` | 359769 | `684d47075e3ed11775a19d7739219e4fdf9d1a3d70cab8de1db97f23a54267e3` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/all_python_scan_final.log` | 891 | `fbbba7f273426782087680d0178a5198ad427695e3e3dc9c2286651300e4e55c` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/check_navdp_matrix.py` | 1620 | `f928700411e34d1b178fb77d1663a42645459501a10d63b10e1d13ee13fac057` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/finalize_sdk_report.py` | 10988 | `f97b3541fb17eb34a2b21e0a5fec0eb6ecca17782623cd2e311d45d73ff9c6a1` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/gdb_cpython_disassembly.txt` | 17546 | `c26f886280b494805b0d2ab131a78e4664798ee887d8015f290c856793435153` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/gdb_hooks.py` | 4123 | `958e82b53cc40eabddc1f963bace7b9f6bb44feec2c16aa2b961f4dad10f9234` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/gdb_native_commands.txt` | 263 | `47cbe479e2193504907455d1d9bea3da11cef6ab75ec89b16144a6a38a0469bd` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/gdb_scene_commands.txt` | 269 | `8e1edbe489b91313c3a0269d53799dd1d2262db9198dd3e6dec77ceb502c857b` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/gdb_scene_hooks.py` | 5178 | `7caef0e0592af0687bbf197254e0d17c28f4035e4ea11c7c0f9fe1fbd057b21c` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/gdb_scene_instances.py` | 7442 | `3db331213ad38b9b4ec2b9e18c020f8603c03569dbb388eb2382ed500ba1fca8` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/gdb_scene_instances_commands.txt` | 268 | `8e3c9c54787b6b12460a3f7a3bf8377a2348abce9bbc7bed53bc2e4a6097f878` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/gdb_subtype_disassembly.txt` | 42127 | `c4917e8fd5955d03a751803fc621d62b39915e446973781dbdba9fac2a8e2c88` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/gdb_type_dict_disassembly.txt` | 2259 | `b75d67647e82e33edd06d6ba37955a2cd25e86437042f6802e7c2cef2c71685b` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/machine_inventory.json` | 347923 | `141bc1df5e1eb9c7e68fa477dfa47fb8fcecb70f8bcf23597196dc62f04c6dd7` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/navdp_live_sdk_cpu.py` | 3331 | `a6cdecadc731f0b169a5f3817d4af78ea5d87aeddca6b3ebd3c74c42c29d1e6e` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/patch_navdp.py` | 6200 | `1eef05e59f3c58f582288ea308121d8e0b6ad963953f7928f18bfe056c1781a6` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/patch_navdp_quaternion.py` | 4970 | `9b666e7b55a800cc97a60ae43256ed5176a7eabc10a243f374f3eae8041c5968` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/patch_vendor_critical.py` | 3978 | `781dac8267fe9861f508aecde5435bfeea1cebbe67501f9fc8f9eb3e15f47b2f` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/preclose_native_release.py` | 1914 | `11c9da95d47d5d94bc70fa004aea681a6a13e52bb26ebd04f1036a47371fbbdb` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/refresh_sdk_final.py` | 11766 | `030a8bcf96759ad02fa6aecc89000f8a50f9c0a6df526a2a328c9b92b7d6c15d` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/ruff_critical.json` | 322144 | `371d4b7a6ca70d48c8ba1effa644e5da30ab8380b86997faf5b70d59cfb7dbf6` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/ruff_critical_after.json` | 311217 | `c760dcd0e8ca76ddf758aa6b0fa179e0a2a360a4245d1460c8152ebd83a9588d` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/run_newton_wrapper_cpu.py` | 2340 | `79fe0e703c9e15005d76d7f28f2d70db73005e5c2232d78b788ca1a24111e03d` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/scan_all_project_python.py` | 4701 | `706751538d085a31ae719d42cdf395aa2c37177b33008f4b79a40d789e44e63f` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/scan_sdk.py` | 6139 | `2f500d2f8890725c06d55e6df90a282d89f48c877a3de6d5dd4566aa71966a74` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/scene_native_gdb.log` | 1094085 | `f37b39054469e1ce8b9880dd84319f3ff28e1ce1e852049c4e9daafd36658518` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/scene_native_gdb_instances.log` | 1148180 | `418e42b7af1b1076cbf3088a311a982b36502c5ce3023db0d1e2d660a8021f48` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/sdk_changed_modules.json` | 592793 | `8afa9d50235f710adbe9445e13d531ceca79ebf6a651e417d07d47ab7044f4d4` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/test_optional_vendor_regressions.py` | 14886 | `7ad145c361676f041908235cff4c7de4ad5e865baea8efc688ab00554223b5b8` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/documentation_reference_inventory.json` | 221093 | `a36032823cd8e2bda4c0e8d072cced885a166c0ed188376852732e6340900fca` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/sdk_audit/documentation_legacy_reference_candidates.json` | 169327 | `5a159cc36528301423bde0f81add34399b2c407d48d9f48b2b211e5947ace89b` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/physics/SecondFloor_N1_60_07_execution.json` | 43431 | `65211a6076bc49914e5203969eb9299c561c9a42e6e8f5ce1242e03200222d76` |
| `/home/lfm/Documents/ChatGPT/Social Navigation/migration_sim61_lab3ea_20261010/physics/SecondFloor_N1_60_07/result.json` | 341142 | `1a39a9f10fcd332750f9b38b08dc0b92273e2bbb1e02768f7c8c5e395968eec6` |
