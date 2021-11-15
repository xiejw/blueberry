EVA_PATH        = ../eva
EVA_LIB         = ${EVA_PATH}/.build_release/libeva.a

MLVM_PATH       = ../mlvm
MLVM_LIB        = ${MLVM_PATH}/.build_release/libmlvm.a

# eva.mk offers ${BUILD}, fmt (action), objs fn.
include ${EVA_PATH}/eva.mk

# ------------------------------------------------------------------------------
# configurations.
# ------------------------------------------------------------------------------

SRC             =  src
CMD             =  cmd
FMT_FOLDERS     =  ${SRC} ${CMD}  # required by eva.mk

CFLAGS          += -I${SRC} -I${EVA_PATH}/src
CFLAGS          += -DVM_SPEC -I${MLVM_PATH}/src -I${MLVM_PATH}/include
LDFLAGS         += ${MLVM_LIB} ${EVA_LIB}

# ------------------------------------------------------------------------------
# libs.
# ------------------------------------------------------------------------------

# ALL_LIBS        = ${BB_LIB}
ALL_LIBS =


# ------------------------------------------------------------------------------
# actions.
# ------------------------------------------------------------------------------

.DEFAULT_GOAL   = compile

compile: ${BUILD} ${ALL_LIBS}

# DEP_FLAGS += -B
#
# ifdef RELEASE
# DEP_FLAGS += RELEASE=1
# endif
#
# ifdef BLIS
# DEP_FLAGS += BLIS=1
# endif

compile_all:
	${MK} -C ../eva ${DEP_FLAGS} libeva
	${MK} -C ../mlvm ${DEP_FLAGS} libmlvm
	${MK} ${DEP_FLAGS}

# ------------------------------------------------------------------------------
# cmds.
# ------------------------------------------------------------------------------

# Put `test` out from CMDS, as it needs special testing library in next section.
CMD_CANDIDATES  = $(patsubst ${CMD}/%,%,$(wildcard ${CMD}/*))
CMDS            = $(filter-out test,${CMD_CANDIDATES})
CMD_TARGETS     = $(patsubst ${CMD}/%/main.c,${BUILD}/%,$(wildcard ${CMD}/*/main.c))

compile: ${CMD_TARGETS}

$(foreach cmd,$(CMDS),$(eval $(call objs,$(cmd),$(BUILD),$(BB_LIB))))


