target_sources(
    path_linalg
    PRIVATE
    runner.cc
    host_minplus.cc
    naive_minplus.cu
    naive_minplus_branch.cu
    naive_minplus_contiguous.cu
    naive_minplus_shared.cu
    naive_minplus_indexing_shared.cu
)