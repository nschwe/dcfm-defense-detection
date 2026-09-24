build_lib(
  LIBNAME iolsr
  SOURCE_FILES
    helper/iolsr-helper.cc
    model/iolsr-header.cc
    model/iolsr-defense-strategy.cc
    model/iolsr-defense-fpnt.cc
    model/iolsr-routing-protocol.cc
    model/iolsr-state.cc
  HEADER_FILES
    helper/iolsr-helper.h
    model/iolsr-header.h
    model/iolsr-defense-strategy.h
    model/iolsr-defense-fpnt.h
    model/iolsr-repositories.h
    model/iolsr-routing-protocol.h
    model/iolsr-state.h
  LIBRARIES_TO_LINK ${libinternet}
)
