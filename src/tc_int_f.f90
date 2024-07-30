
program tc_int

    use gpu_module

    implicit none

    integer :: nBlocks, blockSize
    integer :: n_grid1, n_grid2
    integer :: n_ao
    integer :: n_nuc
    integer :: size_bh

    double precision, allocatable :: r1(:,:), wr1(:), r2(:,:), wr2(:), rn(:,:)
    double precision, allocatable :: aos_data1(:,:,:), aos_data2(:,:,:)
    double precision, allocatable :: c_bh(:,:)
    integer,          allocatable :: m_bh(:,:), n_bh(:,:), o_bh(:,:)
    double precision, allocatable :: int2_grad1_u12_ao(:,:,:,:)
    double precision, allocatable :: int_2e_ao(:,:,:,:)


    nBlocks = 1
    blockSize = 32

    n_grid1 = 1000
    n_grid2 = 10000

    n_ao = 20
    n_nuc = 3

    size_bh = 20

    allocate(r1(n_grid1,3), wr1(n_grid1))
    allocate(r2(n_grid2,3), wr2(n_grid2))
    allocate(rn(n_nuc,3))
    allocate(aos_data1(n_grid1,n_ao,4))
    allocate(aos_data2(n_grid2,n_ao,4))
    allocate(c_bh(size_bh,n_nuc), m_bh(size_bh,n_nuc), n_bh(size_bh,n_nuc), o_bh(size_bh,n_nuc))
    allocate(int2_grad1_u12_ao(n_ao,n_ao,n_grid1,4), int_2e_ao(n_ao,n_ao,n_ao,n_ao))

    r1 = 0.d0
    wr1 = 0.d0
    r2 = 0.d0
    wr2 = 0.d0
    rn = 0.d0
    aos_data1 = 0.d0
    aos_data2 = 0.d0
    c_bh = 0.d0
    m_bh = 1
    n_bh = 1
    o_bh = 1

    int2_grad1_u12_ao = 0.d0
    int_2e_ao = 0.d0

    print *, " start call to tc_int_c"

    call tc_int_c(nBlocks, blockSize,                     &
                  n_grid1, n_grid2, n_ao, n_nuc, size_bh, &
                  r1, wr1, r2,  wr2, rn,                  &
                  aos_data1, aos_data2,                   &
                  c_bh, m_bh, n_bh, o_bh,                 &
                  int2_grad1_u12_ao, int_2e_ao)

    print *, " end tc_int_c"

end program tc_int


