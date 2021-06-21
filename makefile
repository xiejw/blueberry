EVA_PATH        = ../eva
EVA_LIB         = ${EVA_PATH}/.build_release/libeva.a

MLVM_PATH       = ../mlvm
MLVM_LIB        = ${MLVM_PATH}/.build_release/libmlvm.a

include ${EVA_PATH}/eva.mk

# ------------------------------------------------------------------------------
# Configurations.
# ------------------------------------------------------------------------------

SRC             =  src
CMD             =  cmd
FMT_FOLDERS     =  ${SRC} ${CMD}  # required by eva.mk

CFLAGS          += -I${SRC} -I${EVA_PATH}/src -g
CFLAGS          += -DVM_SPEC -I${MLVM_PATH}/src -I${MLVM_PATH}/include
LDFLAGS         += ${MLVM_LIB} ${EVA_LIB}

# ------------------------------------------------------------------------------
# Libs.
# ------------------------------------------------------------------------------
BB_HEADER       = ${SRC}/bb.h
BB_LIB          = ${BUILD}/bb_bb.o ${BUILD}/bb_prog.o ${BUILD}/bb_module.o \
		  ${BUILD}/bb_layers.o ${BUILD}/bb_opt.o \
		  ${BUILD}/opt_fn.o ${BUILD}/opt_td_map.o \
		  ${BUILD}/opt_pass_dce.o \
		  ${BUILD}/opt_pass_math.o

ALL_LIBS        = ${BB_LIB}


# ------------------------------------------------------------------------------
# Actions.
# ------------------------------------------------------------------------------

.DEFAULT_GOAL   = compile

compile: ${BUILD} ${ALL_LIBS}

${BUILD}/bb_%.o: ${SRC}/%.c ${BB_HEADER}
	${EVA_CC} -o $@ -c $<

${BUILD}/opt_%.o: ${SRC}/opt/%.c ${BB_HEADER}
	${EVA_CC} -o $@ -c $<

DEP_FLAGS += -B

ifdef RELEASE
DEP_FLAGS += RELEASE=1
endif

ifdef BLIS
DEP_FLAGS += BLIS=1
endif

compile_all:
	make -C ../eva ${DEP_FLAGS} libeva
	make -C ../mlvm ${DEP_FLAGS} libmlvm
	make ${DEP_FLAGS}

# ------------------------------------------------------------------------------
# Cmd.
# ------------------------------------------------------------------------------

# Put `test` out from CMDS, as it needs special testing library in next section.
CMD_CANDIDATES  = $(patsubst ${CMD}/%,%,$(wildcard ${CMD}/*))
CMDS            = $(filter-out test,${CMD_CANDIDATES})
CMD_TARGETS     = $(patsubst ${CMD}/%/main.c,${BUILD}/%,$(wildcard ${CMD}/*/main.c))

compile: ${CMD_TARGETS}

$(foreach cmd,$(CMDS),$(eval $(call objs,$(cmd),$(BUILD),$(BB_LIB))))

# Special mode to run mnist by checking computation only.
mnist_not_run: ${BUILD}/mnist
	${EX} ${BUILD}/mnist -n

